import logging
import subprocess
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(filename)s: %(levelname)s: %(message)s'
)

def run_command(command: str, oknote: str='', errornote: str='', retry: int=3, sleep_second: int=10, final_raise_exception: bool=False):
    """
    params:
        retry: 重试次数
        final_raise_exception: retry=0依旧报错时是否raise exception
    """
    result = subprocess.run(command, shell=True, capture_output=True) # ignore_security_alert RCE
    if result.returncode == 0:
        logging.info(f"{oknote}: {result.stdout.decode()}")
    else:
        if retry>=1:
            time.sleep(sleep_second)
            run_command(command, f'retry_{oknote}', f'retry_{errornote}', retry-1, sleep_second, final_raise_exception)
        else:
            error_note = f'{errornote}: {result.stderr.decode()}'
            if final_raise_exception:
                raise Exception(error_note)
            else:
                logging.error(error_note) # 打印报错信息
    return None

def dataset_random_split(label, seed=717, train_size=0.8, val_size=0.2):
    assert 0 <= train_size + val_size <= 1, \
        'The sum of valid training set size and validation set size ' \
        'must between 0 and 1 (inclusive).'
    
    node_num = len(label)
    pos_idx = [index for index, value in enumerate(label) if value == 1]
    neg_idx = [index for index, value in enumerate(label) if value == 0]
    
    index = pos_idx + neg_idx
    index = np.random.RandomState(seed).permutation(index)
    train_idx = index[:int(train_size * len(index))]
    val_idx = index[len(index) - int(val_size * len(index)):]
    
    train_mask = np.zeros(node_num, dtype=bool)
    val_mask = np.zeros(node_num, dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    
    return torch.tensor(train_mask), torch.tensor(val_mask), train_idx, val_idx

def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4

    def forward(self, logit, target):
        num_class = logit.size(-1)
        alpha = (
            torch.ones(
                num_class,
            )
            - 0.5
        )
        if alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

        # filter -1
        indices = target >= 0
        logit = logit[indices]
        target = target[indices]

        # assert isinstance(self.alpha,torch.Tensor)\
        alpha = alpha.to(logit.device)

        prob = F.softmax(logit, dim=1)

        if prob.dim() > 2:
            # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            N, C = logit.shape[:2]
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        ori_shp = target.shape
        target = target.view(-1, 1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


class EarlyStopper:
    def __init__(self, start_time, patience=30, model_save_dir=None, model_save_name=None):
        self._save_dir = 'checkpoints'
        if model_save_dir:
            self._save_dir = os.path.join(model_save_dir, self._save_dir)

        if model_save_name:
            self._filename = f'{model_save_name}_early_stop_{start_time}.pth'
        else:
            self._filename = f'early_stop_{start_time}.pth'  
            
        os.makedirs(self._save_dir, exist_ok=True)
        self.save_path = os.path.join(self._save_dir, self._filename)
        logging.info(f'[{self.__class__.__name__}]: Saving model to {self.save_path}')

        self.patience = patience
        self.counter = 0
        self.best_ep = -1
        self.best_score = -1
        self.early_stop = False

    def step(self, score, epoch, model):
        if self.best_score is None:
            self.best_score = score
            self.best_ep = epoch
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience} in epoch {epoch}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_ep = epoch
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.save_path)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.save_path))

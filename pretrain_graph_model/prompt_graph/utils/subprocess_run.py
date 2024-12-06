import logging
import subprocess
import time

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

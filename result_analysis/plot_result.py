import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot(data_path, metrics, savepath):
    data = pd.read_csv(data_path)
    # 筛选所需的数据
    filtered_data = data[['project', 'approach', metrics]].dropna()
    if metrics.endswith('token'):
        filtered_data[metrics] = filtered_data[metrics].apply(lambda x: int(x / 1000))

    # 设置画图风格
    sns.set_style("whitegrid")

    # 初始化图表
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='project', y=metrics, hue='approach', data=filtered_data)

    
    # 添加数据标签
    for p in ax.patches:
        height = p.get_height()  # 获取条形高度
        if not pd.isna(height) and height != 0:  # 排除空值
            label = f'{int(height)}' if isinstance(height, (int, float)) and height == int(height) else f'{height:.2f}'
            ax.text(
                p.get_x() + p.get_width() / 2,  # X坐标
                height + 0.01,  # Y坐标 (稍高于柱子)
                label,  # 数据标签
                ha="center", va="bottom", fontsize=10  # 水平和垂直对齐，字体大小
            )

    # 添加标题和坐标标签
    plt.title(f"{metrics.capitalize()} Comparison by Project and Approach")
    plt.xlabel("Project")
    if metrics.endswith('token'):
        plt.ylabel(metrics.capitalize() + " (K)")
    else:
        plt.ylabel(metrics.capitalize())

    # 显示图例
    plt.legend(title="Approach", loc='center left', bbox_to_anchor=(0,0.3))

    # 保存图表
    plt.savefig(savepath)
    # plt.show()


if __name__ == '__main__':
    # 读取Excel文件中的数据
    data_path = sys.argv[1]
    metrics = sys.argv[2]
    savepath = sys.argv[3] if len(sys.argv) > 3 else "/Users/chenyi/Documents/sag/Final_Project/data/coverage/plot/exp2/"
    savepath = os.path.join(savepath, f'{metrics.replace(" ", "_")}_compare.pdf')

    plot(data_path, metrics, savepath)

#  -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # 百分比
    cm = cm.astype('float')  # 归一化
    #str_cm = cm.astype(np.str).tolist()
    #for row in str_cm:
        #print('\t'.join(row))
    '''
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来,这一个for循环
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) == 0:
                cm[i, j]=0
    '''
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels_name,yticklabels=labels_name,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')
    ax.set_xticklabels(labels_name,rotation=45)
    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    '''
    # 标注百分比信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*100 + 0.5) > 0:
                ax.text(j, i, format(int(cm[i, j]*100 + 0.5) , fmt) + '%',
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    '''
    # 标注类别个数信息
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j]*1 + 0.0) > 0:
                ax.text(j, i, format(int(cm[i, j] + 0.0) , fmt) ,
                        ha="center", va="center",
                        color="white"  if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.title(title)    # 图像标题
    fig = plt.figure()
    fig.savefig('/remote-home/xxh/Code/Code/CE_SET/confusion_matrix_result/HRRP_cm.png', format='png')
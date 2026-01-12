#coding utf-8
import pandas as pd
from matplotlib import pyplot as plt
import os



def plot_acc(dir_name,log_path):
    with open("logs/"+dir_name+'/'+log_path, 'r') as f:
        text = f.read()

    text = text.split('\n')
    text = [x for x in text if x[3:7] == 'top1']
    train = [x.split('为')[-1] for x in text if x[:3] == '训练集']
    validation = [x.split('为')[-1] for x in text if x[:3] == '验证集']
    test = [x.split('为')[-1] for x in text if x[:3] == '测试集']

    train = pd.Series([float(x) for x in train], )
    validation = pd.Series([float(x) for x in validation])
    test = pd.Series([float(x) for x in test])

    df = pd.concat([train, validation, test], axis=1)
    df.columns = ['train', 'validation', 'test']
    plt.xlabel('epochs')
    plt.ylabel('accuracy@1')

    plt.plot(df, )
    plt.legend(['train', 'validation', 'test'])
    plt.title(dir_name+'_'+''.join(log_path.split('.')[:-1]))
    plt.show()

log_dirs = os.listdir('logs/')

for ldir in log_dirs:
    log = os.listdir('logs/'+ldir)
    for j in log:
        pth =  j
        plot_acc(ldir,pth)
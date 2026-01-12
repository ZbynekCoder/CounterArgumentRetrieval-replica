import pandas as pd
from matplotlib import pyplot as plt
with open("without_layernorm_relu_with_hard_negatives_0.02_logs.txt",'r') as f:
    text = f.read()

text = text.split('\n')
text = [x for x in text if x[3:7]=='top1']
train = [x.split('为')[-1] for x in text if x[:3]=='训练集']
validation = [x.split('为')[-1] for x in text if x[:3]=='验证集']
test = [x.split('为')[-1] for x in text if x[:3]=='测试集']

train = pd.Series([float(x) for x in train],)
validation = pd.Series([float(x) for x in validation])
test = pd.Series([float(x) for x in test])

df = pd.concat([train,validation,test],axis=1)
df.columns = ['train','validation','test']
plt.xlabel('epochs')
plt.ylabel('accuracy@1')

plt.plot(df,)
plt.legend(['train','validation','test'])
plt.title('without_layernorm_relu_with_hard_negatives_0.02')
plt.show()


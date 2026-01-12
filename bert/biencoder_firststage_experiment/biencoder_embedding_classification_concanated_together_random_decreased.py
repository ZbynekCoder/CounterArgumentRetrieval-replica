#coding utf8

from utils import logger
from torch import nn
from transformers import BertModel, BertConfig,BertTokenizer
import torch
import os
from dataloader import DataLoader as OurDataLoader
from bert.bertdataloader import ArgumentDataSet
from bert.bertdataloader import trans_to_pairs
import random
from torch.utils.data import DataLoader
from bert.negative_embedding_sampler import BallTreeSearcher,BallTreeEvaluater
import numpy as np
from torch.optim import Adam
current_path = os.path.dirname(__file__)
from torch import nn
from config import training_dir, validation_dir, test_dir

logger_filename = 'logs/concatenated_together_random.log'
save_path = 'model_weights/concatenated_together_random.pth'
with open(logger_filename,'a+') as f:
    f.truncate(0)
os.environ["CUDA_VISIBLE_DEVICES"] ='0,1'

ourdataloader = OurDataLoader(training_dir,  validation_dir,
                        test_dir)
training_df, validation_df, test_df = ourdataloader.to_dataframe()


training_df = trans_to_pairs(training_df)
validation_df = trans_to_pairs(validation_df)
test_df = trans_to_pairs(test_df)
training_df = training_df.dropna()
training_df = training_df.reset_index()[training_df.columns]
training_df['negative_text'] = training_df['counter_text']
validation_df = validation_df.dropna()
validation_df = validation_df.reset_index()[validation_df.columns]
validation_df['negative_text'] = validation_df['counter_text']
test_df = test_df.dropna()
test_df = test_df.reset_index()[test_df.columns]
test_df['negative_text'] = test_df['counter_text']
training_dataset = ArgumentDataSet(training_df)
validation_dataset = ArgumentDataSet(validation_df)
test_dataset = ArgumentDataSet(test_df)
'''dataset构造完成'''


'''先训个biencoder看看情况'''
class BiModel(nn.Module):
    def __init__(self): #此处point，counter都为正样本
        super(BiModel, self).__init__()

        self.model1 = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear1 = nn.Linear(768,128) #sim header
        self.linear2 = nn.Linear(2688,2)



    def forward(self,input_ids,token_type_ids,attention_mask):
        x = self.model1(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        return self.linear1(x[1]),x[1] # 768向量用于进一步计算相似度使用


    def classify_pair(self,emb1_1,emb1_2,emb2_1,emb2_2):
        x1_1 = emb1_1
        x1_2 = emb1_2
        x1_diff = x1_1 - x1_2
        x2_1 = emb2_1
        x2_2 = emb2_2

        x2_diff = x2_1 - x2_2
        x = torch.cat([x1_1,x1_2,torch.abs(x1_diff),x2_1,x2_2,torch.abs(x2_diff)],dim=1)#差别 以及差别的绝对值
        x = self.linear2(x)
        return x

    def get_tokenizer(self):
        return self.tokenizer1



model1 = BiModel().cuda()
tokenizer = model1.get_tokenizer()
device_ids = [0, 1]
model1 = torch.nn.DataParallel(model1, device_ids=device_ids)

#model2 = BiModel().cuda()
model2 = model1
point_sim_embeddings = []
counter_sim_embeddings = []
#point_dissim_embeddings = []
#counter_dissim_embeddings = []
training_prepare_dataloader = DataLoader(training_dataset,batch_size=16,shuffle=False,drop_last=False)
print("开始初始化获取负样本的下标")
with torch.no_grad():
    for point, counter, negative in training_prepare_dataloader:
        point = tokenizer(point, padding=True, truncation=True)
        counter = tokenizer(counter, padding=True, truncation=True)
        point_emb1,point_emb2 = model1(input_ids=torch.tensor(point['input_ids']).cuda(),
                           attention_mask=torch.tensor(point['attention_mask']).cuda(),
                           token_type_ids=torch.tensor(point['token_type_ids']).cuda())
        counter_emb1,counter_emb2 = model2(input_ids=torch.tensor(counter['input_ids']).cuda(),
                             attention_mask=torch.tensor(counter['attention_mask']).cuda(),
                             token_type_ids=torch.tensor(counter['token_type_ids']).cuda())
        point_emb1 = point_emb1.cpu().numpy().tolist()
        counter_emb1 = counter_emb1.cpu().numpy().tolist()
        #point_emb_sim = point_emb_sim.cpu().numpy().tolist()
        #counter_emb_sim = counter_emb_sim.cpu().numpy().tolist()
        point_sim_embeddings.append((point_emb1,point_emb2))
        counter_sim_embeddings.append((counter_emb1,counter_emb2))
        #point_dissim_embeddings.append(point_emb_dissim)
        #counter_dissim_embeddings.append(counter_emb_dissim)
point_sim_embeddings = np.concatenate([x[0] for x in point_sim_embeddings])
counter_sim_embeddings = np.concatenate([x[0] for x in counter_sim_embeddings])
#point_dissim_embeddings = np.concatenate(point_dissim_embeddings)
#counter_dissim_embeddings = np.concatenate(counter_dissim_embeddings)
searcher = BallTreeSearcher(point_sim_embeddings,counter_sim_embeddings)
'''获取负样本下标'''
negative_index = searcher.search(([ i for i in range(point_sim_embeddings.shape[0])],point_sim_embeddings))
point_counter_col = ['point_text','counter_text']
training_df['negative_text'] = [ training_df[point_counter_col[x[0]]].iloc[x[1]] for x in negative_index ]

'''正式开始训练'''
epochs = 200
#optimizer = Adam([{'params':model1.parameters()},{'params':model2.parameters()}],lr=3e-6)
optimizer = Adam(model1.parameters(),lr=3e-6)
#triplet_loss_sim = nn.TripletMarginLoss(margin=1.0, p=2)
#triplet_loss_dissim = nn.TripletMarginLoss(margin=1.0, p=2)

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.tripletloss = nn.TripletMarginLoss(margin=1.0,p=2)
        self.cross_entropy = nn.CrossEntropyLoss()
    def forward(self,anchor,positive,negative,positive_prob,negative_prob):
        loss1 = self.tripletloss(anchor,positive,negative)
        loss2 = self.cross_entropy(positive_prob,torch.ones_like(positive_prob,dtype=torch.int64)[:,0])
        loss3 = self.cross_entropy(negative_prob,torch.zeros_like(negative_prob,dtype=torch.int64)[:,0])
        loss =  loss1+loss2+loss3
        return loss


combinedloss = CombinedLoss()

def byindex(elem):
    return elem[2]

for i in range(epochs):
    print("开始第{}个epoch的训练".format(i))
    training_dataloader = DataLoader(training_dataset, batch_size=8, shuffle=True, drop_last=True)
    zero_count = 0
    for idx,(point, counter, negative) in enumerate(training_dataloader):
        point = tokenizer(point, padding=True, truncation=True)
        counter = tokenizer(counter, padding=True, truncation=True)
        negative = tokenizer(negative, padding=True, truncation=True)
        point_emb1,point_emb2 = model1(input_ids=torch.tensor(point['input_ids']).cuda(),
                           attention_mask=torch.tensor(point['attention_mask']).cuda(),
                           token_type_ids=torch.tensor(point['token_type_ids']).cuda())
        counter_emb1,counter_emb2 = model2(input_ids=torch.tensor(counter['input_ids']).cuda(),
                             attention_mask=torch.tensor(counter['attention_mask']).cuda(),
                             token_type_ids=torch.tensor(counter['token_type_ids']).cuda())
        negative_emb1,negative_emb2 = model2(input_ids=torch.tensor(negative['input_ids']).cuda(),
                             attention_mask=torch.tensor(negative['attention_mask']).cuda(),
                             token_type_ids=torch.tensor(negative['token_type_ids']).cuda())
        positive_pair = model1.module.classify_pair(point_emb1,counter_emb1,point_emb2,counter_emb2)
        negative_pair = model1.module.classify_pair(point_emb1,negative_emb1,point_emb2,negative_emb2)
        #random_flag = torch.tensor(random_flag).unsqueeze(dim=1).cuda()
        loss = combinedloss(point_emb1,counter_emb1,negative_emb1,positive_pair,negative_pair)
        if(idx%10==0):
            print("当前第{}个mini batch，当前loss为{}".format(idx,loss))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("开始获取负样本的下标")
    point_sim_embeddings = []
    counter_sim_embeddings = []
    training_prepare_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=False, drop_last=False)
    with torch.no_grad():
        for point, counter, negative in training_prepare_dataloader:
            point = tokenizer(point, padding=True, truncation=True)
            counter = tokenizer(counter, padding=True, truncation=True)
            point_emb1,point_emb2 = model1(input_ids=torch.tensor(point['input_ids']).cuda(),
                               attention_mask=torch.tensor(point['attention_mask']).cuda(),
                               token_type_ids=torch.tensor(point['token_type_ids']).cuda())
            counter_emb1,counter_emb2 = model2(input_ids=torch.tensor(counter['input_ids']).cuda(),
                                 attention_mask=torch.tensor(counter['attention_mask']).cuda(),
                                 token_type_ids=torch.tensor(counter['token_type_ids']).cuda())
            point_emb1 = point_emb1.cpu().numpy().tolist()
            counter_emb1 = counter_emb1.cpu().numpy().tolist()
            point_emb2 = point_emb2.cpu().numpy().tolist()
            counter_emb2 = counter_emb2.cpu().numpy().tolist()
            point_sim_embeddings.append((point_emb1,point_emb2))
            counter_sim_embeddings.append((counter_emb1,counter_emb2))
    point_sim_embeddings_ = np.concatenate([x[0] for x in point_sim_embeddings])
    counter_sim_embeddings_ = np.concatenate([x[0] for x in counter_sim_embeddings])
    searcher = BallTreeSearcher(point_sim_embeddings_, counter_sim_embeddings_)
    '''获取负样本下标'''
    negative_index = searcher.search(([i for i in range(point_sim_embeddings_.shape[0])], point_sim_embeddings_),
                                     0.8-i*0.02 if 0.8-i*0.02>0 else 0 )

    point_counter_col = ['point_text', 'counter_text']
    training_df['negative_text'] = [ training_df[point_counter_col[x[0]]].iloc[x[1]] for x in negative_index ]

    evaluater = BallTreeEvaluater(point_sim_embeddings,counter_sim_embeddings,model1)
    #point_sim_embeddings = np.concatenate([x[1] for x in point_sim_embeddings])
    accuracy_at_top1 = evaluater.cal_accuracy(point_sim_embeddings_,10)
    print("训练集top1 accuracy为{}".format(accuracy_at_top1))
    logger("训练集top1 accuracy为{}".format(accuracy_at_top1),logger_filename)
    point_sim_embeddings = []
    counter_sim_embeddings = []

    validation_prepare_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False, drop_last=False)
    with torch.no_grad():
        for point, counter, negative in validation_prepare_dataloader:
            point = tokenizer(point, padding=True, truncation=True)
            counter = tokenizer(counter, padding=True, truncation=True)
            point_emb1,point_emb2= model1(input_ids=torch.tensor(point['input_ids']).cuda(),
                               attention_mask=torch.tensor(point['attention_mask']).cuda(),
                               token_type_ids=torch.tensor(point['token_type_ids']).cuda())
            counter_emb1,counter_emb2 = model2(input_ids=torch.tensor(counter['input_ids']).cuda(),
                                 attention_mask=torch.tensor(counter['attention_mask']).cuda(),
                                 token_type_ids=torch.tensor(counter['token_type_ids']).cuda())
            point_emb1 = point_emb1.cpu().numpy().tolist()
            counter_emb1 = counter_emb1.cpu().numpy().tolist()
            point_emb2 = point_emb2.cpu().numpy().tolist()
            counter_emb2 = counter_emb2.cpu().numpy().tolist()
            point_sim_embeddings.append((point_emb1,point_emb2))
            counter_sim_embeddings.append((counter_emb1,counter_emb2))
    point_sim_embeddings_ = np.concatenate([x[0] for x in point_sim_embeddings])
    counter_sim_embeddings_ = np.concatenate([x[0] for x in counter_sim_embeddings])

    #searcher = BallTreeSearcher(point_embeddings, counter_embeddings)
    #'''获取负样本下标'''
    #negative_index = searcher.search(([i for i in range(point_embeddings.shape[0])], point_embeddings))

    point_counter_col = ['point_text', 'counter_text']
    #training_df['negative_text'] = [ training_df[point_counter_col[x[0]]].iloc[x[1]] for x in negative_index ]
    #point_sim_embeddings = np.concatenate([x[1] for x in point_sim_embeddings])
    evaluater = BallTreeEvaluater(point_sim_embeddings,counter_sim_embeddings,model1)
    #point_sim_embeddings = [x[1] for x in point_sim_embeddings]
    accuracy_at_top1 = evaluater.cal_accuracy(point_sim_embeddings_,10)
    print("验证集top1 accuracy为{}".format(accuracy_at_top1))
    logger("验证集top1 accuracy为{}".format(accuracy_at_top1),logger_filename)
    point_sim_embeddings = []
    counter_sim_embeddings = []

    test_prepare_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False)
    with torch.no_grad():
        for point, counter, negative in test_prepare_dataloader:
            point = tokenizer(point, padding=True, truncation=True)
            counter = tokenizer(counter, padding=True, truncation=True)
            point_emb1,point_emb2= model1(input_ids=torch.tensor(point['input_ids']).cuda(),
                               attention_mask=torch.tensor(point['attention_mask']).cuda(),
                               token_type_ids=torch.tensor(point['token_type_ids']).cuda())
            counter_emb1,counter_emb2 = model2(input_ids=torch.tensor(counter['input_ids']).cuda(),
                                 attention_mask=torch.tensor(counter['attention_mask']).cuda(),
                                 token_type_ids=torch.tensor(counter['token_type_ids']).cuda())
            point_emb1 = point_emb1.cpu().numpy().tolist()
            counter_emb1 = counter_emb1.cpu().numpy().tolist()
            point_emb2 = point_emb2.cpu().numpy().tolist()
            counter_emb2 = counter_emb2.cpu().numpy().tolist()
            point_sim_embeddings.append((point_emb1,point_emb2))
            counter_sim_embeddings.append((counter_emb1,counter_emb2))
    point_sim_embeddings_ = np.concatenate([x[0] for x in point_sim_embeddings])
    counter_sim_embeddings_ = np.concatenate([x[0] for x in counter_sim_embeddings])

    #searcher = BallTreeSearcher(point_embeddings, counter_embeddings)
    #'''获取负样本下标'''
    #negative_index = searcher.search(([i for i in range(point_embeddings.shape[0])], point_embeddings))

    point_counter_col = ['point_text', 'counter_text']
    #training_df['negative_text'] = [ training_df[point_counter_col[x[0]]].iloc[x[1]] for x in negative_index ]
    #point_sim_embeddings = np.concatenate([x[1] for x in point_sim_embeddings])
    evaluater = BallTreeEvaluater(point_sim_embeddings,counter_sim_embeddings,model1)
    #point_sim_embeddings = [x[1] for x in point_sim_embeddings]
    accuracy_at_top1 = evaluater.cal_accuracy(point_sim_embeddings_,10)
    print("测试集top1 accuracy为{}".format(accuracy_at_top1))
    logger("测试集top1 accuracy为{}".format(accuracy_at_top1), logger_filename)
model1 = model1.cpu()
torch.save(model1, save_path)


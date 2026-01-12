#coding: utf-8
from transformers import BertTokenizer,BertModel
from bert.negative_embedding_sampler import BallTreeEvaluater
from utils import group_and_split
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.neighbors import BallTree
import numpy as np

'''用模型和get_embedding来进行获得dataframe的各行向量'''

'''使用函数构建分组'''
def get_tasks_data(df):
    result = dict()
    result['sdoc'] = group_and_split(df,'sdoc')
    result['sdoa'] = group_and_split(df,'sdoa')
    result['sdc'] = group_and_split(df,'sdc')
    result['sda'] = group_and_split(df,'sda')
    result['stc'] = group_and_split(df,'stc')
    result['sta'] = group_and_split(df,'sta')
    result['epc'] = group_and_split(df,'epc')
    result['epa'] = group_and_split(df,'epa')
    return result

class ArgumentDataSet(Dataset):
    def __init__(self, df):
        # def __init__(self,input_ids,attention_mask,label):
        self.df = df
        self.text = df['text']


    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, index):
        return self.text[index]

def extract_embedding(df,model,batch_size):
    tokenizer = model.tokenizer1
    dataset = ArgumentDataSet(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    embedding1 = []
    embedding2 = []
    for x in dataloader:
        point = tokenizer(x, padding=True, truncation=True)
        point_emb1, point_emb2 = model(input_ids=torch.tensor(point['input_ids']).cuda(),
                                    attention_mask=torch.tensor(point['attention_mask']).cuda(),
                                    token_type_ids=torch.tensor(point['token_type_ids']).cuda())
        point_emb1 = point_emb1.cpu().numpy().tolist()
        point_emb2 = point_emb2.cpu().numpy().tolist()
        embedding1 = embedding1 + point_emb1
        embedding2 = embedding2 + point_emb2
    df['embedding1'] = embedding1
    df['embedding2'] = embedding2
    return df


'''构建balltree获取组内最相似样本对，然后模型对最相似样本对进行排序'''
class BallTreeEvaluater(object):
    def __init__(self,counter_emb_candidates,embedding1,embedding2,point_num,classify_pair_layer):
        self.point_num = point_num
        #候选待查向量,构建kd树 用于查询k近邻
        self.sim_embeddings = counter_emb_candidates
        #候选计算距离向量
        self.distance_embeddings1 = embedding1
        self.distance_embeddings2 = embedding2
        self.balltree = BallTree(self.sim_embeddings,leaf_size=2) #默认欧几里得距离
        self.classify_pair_layer = classify_pair_layer

    def cal_tp_at_top1_count(self,point_emb,topk=10):
        '''(index,embedding)形式数据为进来 index用于免得搜到自身，embedding用于搜到的特征'''
        total = self.point_num
        positive = 0
        point_sim = point_emb
        point_dist,point_ind = self.balltree.query(point_sim,k=topk)
        prob = 0.0
        for index_ in range(total): #遍历各个index
            for i in point_ind[index_]: #查询到的向量
                prob_tmp = self.get_prob(self.distance_embeddings1[index_], self.distance_embeddings1[i],
                                         self.distance_embeddings2[index_], self.distance_embeddings2[i])
                if(prob_tmp>prob):
                    prob = prob_tmp
                    topid = i
            if topid==index_:
                positive+=1
            prob = 0.0
        return positive  #* 1.0 / total

    def get_prob(self,point_emb1,counter_emb1,point_emb2,counter_emb2):
        merge_layer = self.classify_pair_layer
        point_emb1 = torch.Tensor(point_emb1).cuda()
        counter_emb1 = torch.Tensor(counter_emb1).cuda()
        point_emb2 = torch.Tensor(point_emb2).cuda()
        counter_emb2 = torch.Tensor(counter_emb2).cuda()
        merged_embedding = merge_layer(point_emb1.unsqueeze(dim=0),counter_emb1.unsqueeze(dim=0),point_emb2.unsqueeze(dim=0),counter_emb2.unsqueeze(dim=0))
        softmax = torch.nn.Softmax(dim=1)
        merged_embedding = softmax(merged_embedding)
        #print(merged_embedding)
        prob = merged_embedding[0][1].cpu().item()  #是正样本的概率大小
        return prob
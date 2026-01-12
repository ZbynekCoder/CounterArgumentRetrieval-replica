import numpy as np
import time
import random

import torch
from sklearn.neighbors import BallTree
import numpy as np
'''
1.用来构建球树，分两部分 一个是 point的球树,一个是counter的球树，point球树取top2，自身肯定不是候选，所以一定是the second，
counter取top2，确认top1是postive则取the second，否则 the first.
2.
'''
class BallTreeSearcher(object):
    def __init__(self,point_emb,counter_emb):
        self.point_emb = point_emb
        self.counter_emb = counter_emb
        self.point_balltree = BallTree(point_emb,leaf_size=2)
        self.counter_balltree = BallTree(counter_emb,leaf_size=2)

    def search(self,point_emb,random_rate=0.8):
        '''(index,embedding)形式数据为进来 index用于免得搜到自身，embedding用于搜到的特征'''
        point_index = point_emb[0]
        point_emb = point_emb[1]
        point_dist,point_ind = self.point_balltree.query(point_emb,k=3)
        counter_dist,counter_ind = self.counter_balltree.query(point_emb,k=3)
        negative_index = []
        for idx,index_ in enumerate(point_index): #遍历各个index
            '''index等于自己的不取'''
            if(index_==point_ind[idx][0]):
                point_index_ = 1 #不要0 要1
            else:
                point_index_ = 0

            if(index_==counter_ind[idx][0]):
                counter_index_ = 1 #不要0 要1
            else:
                counter_index_ = 0
            '''比较point和counter,选出距离最小的,由此选定向量'''
            if(random.random()>random_rate): #比该数小的是随机的，，因此起到随机比率作用
                if (point_dist[idx][point_index_] > counter_dist[idx][counter_index_]):
                    negative_index.append((1, counter_ind[idx][counter_index_]))
                else:
                    negative_index.append((0, point_ind[idx][point_index_]))
            else:
                negative_index.append((random.randint(0,1), random.randint(0,len(point_index)-1)))
        return negative_index

class BallTreeEvaluater(object):
    def __init__(self,point_sim_embeddings,counter_sim_embeddings,model):
        self.point_num = len(point_sim_embeddings)
        self.sim_embeddings = np.concatenate([np.concatenate([x[0] for x in counter_sim_embeddings]),
                                              np.concatenate([x[0]for x in point_sim_embeddings])],axis=0)
        #here modify it
        self.distance_embeddings1 = np.concatenate([np.concatenate([x[0] for x in counter_sim_embeddings]),
                                              np.concatenate([x[0] for x in point_sim_embeddings])],axis=0)
        self.distance_embeddings2 = np.concatenate([np.concatenate([x[1] for x in counter_sim_embeddings]),
                                              np.concatenate([x[1] for x in point_sim_embeddings])],axis=0)
        self.balltree = BallTree(self.sim_embeddings,leaf_size=2) #默认欧几里得距离
        self.model = model

    def cal_accuracy(self,point_emb,topk=10):
        '''(index,embedding)形式数据为进来 index用于免得搜到自身，embedding用于搜到的特征'''
        total = self.point_num
        positive = 0
        point_sim = point_emb
        point_dist,point_ind = self.balltree.query(point_sim,k=topk)
        prob = 0.0
        for index_ in range(total): #遍历各个index
            for i in point_ind[index_]:
                prob_tmp = self.get_prob(self.distance_embeddings1[index_], self.distance_embeddings1[i],
                                         self.distance_embeddings2[index_], self.distance_embeddings2[i])
                if(prob_tmp>prob):
                    prob = prob_tmp
                    topid = i
            if topid==index_:
                positive+=1
            prob = 0.0
        return positive * 1.0 / total

    def get_prob(self,point_emb1,counter_emb1,point_emb2,counter_emb2):
        merge_layer = self.model.module.classify_pair
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

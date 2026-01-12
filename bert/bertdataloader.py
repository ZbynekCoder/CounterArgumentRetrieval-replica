from dataloader import DataLoader as OurDataloader
from config import training_dir, validation_dir, test_dir
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm

import os
current_path = os.path.dirname(__file__)


def trans_to_pairs(df):
    df = df.groupby(['domain','argumentation_title','utterence_id','stance'])
    df = [x for x in df]
    result = []
    for x in df:
        cur = {}
        cur['domain'] = x[0][0]
        cur['title'] = x[0][1]
        cur['stance'] = x[0][3]
        for i in x[1].iterrows():
            cur[i[1]['utterence_type']+'_text'] = i[1]['text']
        result.append(cur)
    result = pd.DataFrame(result)
    return result




class ArgumentDataSet(Dataset):
    def __init__(self, df):
        # def __init__(self,input_ids,attention_mask,label):
        self.df = df
        self.point = df['point_text']
        self.counter = df['counter_text']
        self.negative_text = df['negative_text']


    def __len__(self, ):
        return len(self.df)

    def __getitem__(self, index):
        return self.point[index],self.counter[index],self.negative_text[index]

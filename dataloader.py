# coding:utf-8
import os

import pandas

from config import training_dir, validation_dir, test_dir
import pandas as pd
from tqdm import tqdm
current_path = os.path.dirname(__file__)

class DataLoader(object):
    def __init__(self, training_dir, validation_dir, test_dir):
        self.training_dir = current_path+'/'+training_dir
        self.validation_dir = current_path+'/'+validation_dir
        self.test_dir = current_path+'/'+test_dir
        self.training_files = os.listdir(self.training_dir)
        self.validation_files = os.listdir(self.validation_dir)
        self.test_files = os.listdir(self.test_dir)
        try:
            self.training_files.remove('.DS_Store')
            self.validation_files.remove('.DS_Store')
            self.test_files.remove('.DS_Store')
        except:
            pass
        self.dirs = {'training': self.training_dir, 'validation': self.validation_dir, 'test': self.test_dir}
        self.training_data = self.read_data(dtype='training')
        self.validation_data = self.read_data(dtype='validation')
        self.test_data = self.read_data('test')

    def read_data(self,dtype='training'): # training,validation,test三类
        argumentation_list = []
        data_dir = self.dirs[dtype]
        topics = os.listdir(data_dir)
        try:
            topics.remove('.DS_Store')
        except:
            pass
        for topic in topics:
            topic_dir = data_dir + topic + '/'
            argumentations = os.listdir(topic_dir)
            if '.DS_Store' in argumentations:
                argumentations.remove('.DS_Store')
            for argumentation in argumentations:
                argumentation_list.append(topic_dir+argumentation)
        #print(len(argumentation_list))
        pro_files = []
        con_files = []
        for argumentation in argumentation_list:
            pro_file = os.listdir(argumentation+'/pro')
            for f in pro_file:
                pro_files.append(argumentation+'/pro/'+f)
            con_file = os.listdir(argumentation + '/con')
            for f in con_file:
                con_files.append(argumentation + '/con/' + f)
        #print(pro_files+con_files)
        files = pro_files + con_files
        data = []
        print("读取并整理数据集\n")
        for file in tqdm(files):
            #print(file)
            domain,argumentation_title,stance = file.split('/')[-4:-1]
            #print(domain,argumentation_title,stance)
            utterence_id, utterence_type = file.split('/')[-1][:-4].split('-')
            with open(file,'r',encoding='utf8') as f:
                text = f.read()

            data.append({'domain':domain,'argumentation_title':argumentation_title,'stance':stance,'utterence_id':utterence_id[:-1],'utterence_type':utterence_type,'text':text})
        print('\n')
        return data

    def to_dataframe(self):
        training_df = pandas.DataFrame.from_dict(self.training_data)
        validation_df = pandas.DataFrame.from_dict(self.validation_data)
        test_df = pandas.DataFrame.from_dict(self.test_data)
        return training_df,validation_df,test_df





if __name__ == '__main__':
    dataset = DataLoader(training_dir,validation_dir,test_dir)
    training_df,validation_df,test_df = dataset.to_dataframe()
    print(training_df[['utterence_id','utterence_type']])
    print(training_df.columns)
    #print([x for x in training_df.groupby('domain')])

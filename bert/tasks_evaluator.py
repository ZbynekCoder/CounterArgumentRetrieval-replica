#coding: utf-8
from transformers import BertTokenizer
from bert.negative_embedding_sampler import BallTreeEvaluater
from utils import group_and_split

'''在8个任务上进行测试'''
tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')


'''获取训练好的句子向量'''
def get_embeding(text,model):
    tokenized = tokenizer1([text])
    input_ids = tokenized['input_ids'].cuda()
    attention_mask = tokenized['attention_mask'].cuda()
    token_type_ids = tokenized['token_type_ids'].cuda()
    embedding1,embedding2 = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
    embedding1 = embedding1.cpu().numpy().tolist()[0]
    embedding2 = embedding2.cpu().numpy().tolist()[0]
    return embedding1,embedding2

'''把模型加载进来'''


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
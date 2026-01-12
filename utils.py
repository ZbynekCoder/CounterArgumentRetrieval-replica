#coding: utf-8


def split_point_counter_for_opposing_debate(data):
    data_grouped = []
    for df in data:
        point = []
        counter = []
        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point':
                point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[(df['utterence_id']==id_)&(df['utterence_type']=='counter')&(df['stance']==row['stance'])]
                if len(row_)==1:
                    counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    point.pop()

        data_grouped.append({'point': point, 'counter': counter})
    return data_grouped


def split_point_counter_for_debate(data):
    data_grouped = []
    '''划分为pro、con两块'''
    for df in data:
        pro_point = []
        pro_counter = []
        con_point = []
        con_counter = []
        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point' and row['stance'] == 'pro':
                pro_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[(df['utterence_id']==id_)&(df['utterence_type']=='counter')&(df['stance']==row['stance'])]
                if len(row_)==1:
                    pro_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    pro_point.pop()

            if row['utterence_type'] == 'point' and row['stance'] == 'con':
                con_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[(df['utterence_id']==id_)&(df['utterence_type']=='counter')&(df['stance']==row['stance'])]
                if len(row_)==1:
                    con_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    con_point.pop()

        pro_counter_ = pro_counter + con_counter
        con_counter_ = con_counter + pro_counter
        data_grouped.append({'point': pro_point, 'counter': pro_counter_})
        data_grouped.append({'point': con_point, 'counter': con_counter_})
    return data_grouped


def split_point_opposing_argument_for_debate(data):
    data_grouped = []
    for df in data:
        pro_point = []
        pro_counter = []
        con_point = []
        con_counter = []

        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point' and row['stance']=='pro':
                pro_point.append(row)
                flag = True  # 未找到的标记
                id_ = row['utterence_id']
                row_ = df.loc[(df['utterence_id']==id_)&(df['utterence_type']=='counter')&(df['stance']==row['stance'])]
                if len(row_)==1:
                    pro_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    pro_point.pop()

            elif row['utterence_type'] == 'point' and row['stance']=='con':
                con_point.append(row)
                id_ = row['utterence_id']
                flag = True  # 未找到的标记
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])]
                if len(row_)==1:
                    con_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    con_point.pop()

        pro_argument = pro_counter + con_point
        con_argument = con_counter + pro_point
        data_grouped.append({'point': pro_point, 'argument': pro_argument})
        data_grouped.append({'point': con_point, 'argument': con_argument})
    return data_grouped


def split_point_counter_for_theme(data):
    data_grouped = []
    '''划分为pro、con两块'''

    #print(data[0][['stance','utterence_type']].value_counts())
    for df in data:
        pro_point = []
        pro_counter = []
        con_point = []
        con_counter = []
        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point' and row['stance'] == 'pro':
                pro_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])& (df['domain'] == row['domain'])& (df['argumentation_title'] == row['argumentation_title'])]
                if len(row_)==1:
                    pro_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了

                if flag:
                    pro_point.pop()

            if row['utterence_type'] == 'point' and row['stance'] == 'con':
                con_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])& (df['domain'] == row['domain'])& (df['argumentation_title'] == row['argumentation_title'])]
                if len(row_)==1:
                    con_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    con_point.pop()

        pro_counter_ = pro_counter + con_counter
        con_counter_ = con_counter + pro_counter
        data_grouped.append({'point': pro_point, 'counter': pro_counter_})
        data_grouped.append({'point': con_point, 'counter': con_counter_})

    return data_grouped



def split_point_argument_for_theme(data):
    data_grouped = []
    for df in data:
        point = []
        counter = []

        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point':
                point.append(row)
                id_ = row['utterence_id']
                flag = True
                row_ = df.loc[(df['utterence_id']==id_)&(df['utterence_type']=='counter')&(df['stance']==row['stance'])&(df['argumentation_title']==row['argumentation_title'])]
                if len(row_)==1:
                    counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    point.pop()
        argument = counter+point
        data_grouped.append({'point': point, 'argument': argument})
    return data_grouped



def split_point_argument_for_debate(data):
    data_grouped = []
    for df in data:
        point = []
        counter = []

        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point':
                point.append(row)
                id_ = row['utterence_id']
                flag = True
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])]
                if len(row_)==1:
                    counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了

                if flag:
                    point.pop()

        argument = counter+point
        data_grouped.append({'point': point, 'argument': argument})
    return data_grouped


def split_point_counter_for_entire_portal(data):
    data_grouped = []
    '''划分为pro、con两块'''
    for df in data:
        pro_point = []
        pro_counter = []
        con_point = []
        con_counter = []
        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point' and row['stance'] == 'pro':
                pro_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])& (df['domain'] == row['domain'])& (df['argumentation_title'] == row['argumentation_title'])]
                if len(row_)==1:
                    pro_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了

                if flag:
                    pro_point.pop()
            if row['utterence_type'] == 'point' and row['stance'] == 'con':
                con_point.append(row)
                id_ = row['utterence_id']
                flag = True #未找到的标记
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])& (df['domain'] == row['domain'])& (df['argumentation_title'] == row['argumentation_title'])]
                if len(row_)==1:
                    con_counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    con_point.pop()
        pro_counter_ = pro_counter + con_counter
        con_counter_ = con_counter + pro_counter
        data_grouped.append({'point': pro_point, 'counter': pro_counter_})
        data_grouped.append({'point': con_point, 'counter': con_counter_})
    return data_grouped



def split_point_argument_for_entire_portal(data):
    data_grouped = []
    for df in data:
        point = []
        counter = []

        for idx, row in df.iterrows():
            if row['utterence_type'] == 'point':
                point.append(row)
                id_ = row['utterence_id']
                flag = True
                row_ = df.loc[
                    (df['utterence_id'] == id_) & (df['utterence_type'] == 'counter') & (df['stance'] == row['stance'])& (df['domain'] == row['domain'])& (df['argumentation_title'] == row['argumentation_title'])]
                if len(row_)==1:
                    counter.append(row_.iloc[0])
                    flag = False #若不存在，把point也给扔了
                if flag:
                    point.pop()
        argument = counter+point
        data_grouped.append({'point': point, 'argument': argument})
    return data_grouped



def group_method(data,group_level):
    level = {
        'sdoc':[ 'domain', 'argumentation_title', 'stance', ],
        'sdoa':[ 'domain', 'argumentation_title', ],
        'sdc':[ 'domain', 'argumentation_title', ],
        'sda':[ 'domain', 'argumentation_title', ],
        'stc':[ 'domain', ],
        'sta':[ 'domain', ],
        'epc':[],
        'epa':[],
    }
    if(len(level[group_level])>0):
        data_grouped = [x for idx, x in data.groupby(level[group_level])]
    else:
        data_grouped = [data, ]
    return data_grouped


def split_method(data,group_level):
    level = {
        'sdoc':split_point_counter_for_opposing_debate,
        'sdoa':split_point_opposing_argument_for_debate,
        'sdc':split_point_counter_for_debate,
        'sda':split_point_argument_for_debate,
        'stc':split_point_counter_for_theme,
        'sta':split_point_argument_for_theme,
        'epc':split_point_counter_for_entire_portal,
        'epa':split_point_argument_for_entire_portal,
    }
    data_group = level[group_level](data)
    return data_group


def group_and_split(data,group_level):
    data = group_method(data,group_level)
    data = split_method(data,group_level)
    return data

def logger(loginfo,filename):
    with open(filename,'a+',) as f:
        f.write(loginfo)
        f.write('\n')




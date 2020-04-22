"""
description text转换基于word2vec
"""

import gensim
import pymysql
import numpy as np
from sklearn import preprocessing
import time


"""对所有description进行embedding"""
def description_total_embedding (description):
    totalembeddings=[]
    for each in description:
        totalembeddings.append(description_embedding(each))
    return totalembeddings


"""对description进行embedding"""
def description_embedding (description):
    token_list = description.split(' ')
    description_vector = np.zeros(300)
    length = 0
    for token in token_list:
        try:
            word_embedding = model[token]
        except KeyError:
            print(token+" not in vocabulary")
            continue
        description_vector += word_embedding
        length += 1
    # 词向量求平均
    if length != 0:
        description_vector = description_vector/length
    return description_vector



""" 训练模型用 """
if __name__ == "__main__":
    start = time.time()
    """数据库连接"""
    conn = pymysql.connect("localhost", "root", "baobei007", "hurricane_data")
    cursor = conn.cursor()
    cursor.execute('select * from users_sample_three_classes_institution_normalized')
    newresults = cursor.fetchall()

    """获取数据"""
    user_id = []
    name = []
    screen_name = []
    namesim = []
    verified = []
    listed_count = []
    friends_count = []
    followers_count = []
    description = []
    label = []
    # character = []
    length = len(newresults)
    for i in range(length):
        user_id.append(newresults[i][0])
        name.append(newresults[i][1])
        screen_name.append(newresults[i][2])
        namesim.append(newresults[i][3])
        verified.append(newresults[i][4])
        listed_count.append(newresults[i][5])
        friends_count.append(newresults[i][6])
        followers_count.append(newresults[i][7])
        description.append(newresults[i][8])
        label.append(newresults[i][9])


    # 对description进行embedding
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    description = description_total_embedding(description)


    """将所有特征进行连接"""
    features_all = np.column_stack([name, screen_name, namesim, friends_count, followers_count, listed_count, verified, description])

    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    features_scale = min_max_scaler.fit_transform(features_all)

    # 数据保存
    np.save('label', label)
    np.save('feature', features_scale)

    end = time.time()
    print('总时间为：', end - start)


"""
word2vec+预测
"""

from numpy import *
import gensim
import pymysql
import math
import pickle
import numpy as np
from sklearn import preprocessing
import time


# 以二进制方式读word2vec模型
model_w2v = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# 以读二进制的方式打开文件
file1 = open("model_three_classes_institution.pickle", "rb")
# 把模型从文件中读取出来
model_predict = pickle.load(file1)
# 关闭文件
file1.close()


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
            word_embedding = model_w2v[token]
        except KeyError:
            print(token+" not in vocabulary")
            continue
        description_vector += word_embedding
        length += 1
    # 词向量求平均
    if length != 0:
        description_vector = description_vector/length
    return description_vector


""" 做预测用 """
def Prediction(results):
    conn1 = pymysql.connect("localhost", "root", "baobei007", "hurricane_data")
    cursor1 = conn1.cursor()
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
    length = len(results)
    for i in range(length):
        user_id.append(results[i][0])
        name.append(results[i][1])
        screen_name.append(results[i][2])
        namesim.append(results[i][3])
        verified.append(results[i][4])
        listed_count.append(results[i][5])
        friends_count.append(results[i][6])
        followers_count.append(results[i][7])
        description.append(results[i][8])
        label.append(results[i][9])

    """对description进行embedding"""
    print('embedding begin:' + str(time.time()))
    description = description_total_embedding(description)

    """将所有特征进行连接"""
    features_all = np.column_stack(
            [name, screen_name, namesim, friends_count, followers_count, listed_count, verified, description])

    # 归一化处理
    min_max_scaler = preprocessing.MinMaxScaler()
    features_scale = min_max_scaler.fit_transform(features_all)

    # 用模型进行预测
    print('prediction begin:' + str(time.time()))
    pred = model_predict.predict(features_scale)
    Label = pred

    # 把预测结果传入数据库
    for i in range(length):
        cursor1.execute("INSERT INTO users_predict_three_classes_institution(user_id,label) VALUES (%s,%s)",(user_id[i], int(Label[i])))

    cursor1.close()
    conn1.commit()
    conn1.close()


if __name__ == "__main__":
    start = time.time()

    """数据库连接"""
    allData = 29388
    dataOfEach = 29388
    batch = math.ceil(allData / dataOfEach)
    BATctrl = 0
    while BATctrl < batch:
        conn2 = pymysql.connect("localhost", "root", "baobei007", "hurricane_data")
        cursor2 = conn2.cursor()
        sql="select * from users_normalized_three_classes_institution limit "+str(dataOfEach * BATctrl)+","+str(dataOfEach)
        cursor2.execute(sql)
        results = cursor2.fetchall()
        results = list(results)
        print('begin from:' + str(dataOfEach * BATctrl))
        Prediction(results)
        BATctrl += 1
        cursor2.close()
        conn2.commit()
        conn2.close()

    end = time.time()
    print('总时间为：', end - start)
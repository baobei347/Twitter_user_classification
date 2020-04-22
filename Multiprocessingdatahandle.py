"""
除description以外的数据预处理
"""
import pymysql
from numpy import *
import codecs
import re
import jieba
from difflib import SequenceMatcher
import time
from multiprocessing import Pool,Manager
import math
import multiprocessing

def Intrans(str): #实现数值转化
    length = len(str)
    return length

def namesimilarize(name,screen_name):#计算单词之间相似度
    name = name.replace(' ','').lower()
    screen_name = screen_name.replace(' ', '').lower()
    s = SequenceMatcher(None, name, screen_name)
    similarank = 0
    ratio = s.ratio()
    if ratio <= 0.7:
        similarank = 1
    else:
        if ratio <= 0.85:
            similarank = 2
        else:
           similarank = 3
    return similarank

def deleteunusefulchar(setence): #删除无用的字符
    int_findall = re.findall('\d{1}', setence)
    m_findall = re.findall('https://t.co/[A-Za-z0-9]+', setence)
    if m_findall:
        for eachone in m_findall:
            eachstring = setence.replace(eachone, '')
            setence = eachstring
    if int_findall:
        for eachone in int_findall:
            eachstring = setence.replace(eachone, ' ')
            setence = eachstring
    splitsetence = jieba.cut(setence, cut_all=False)
    splitoutcome = ''
    for eachsplit in splitsetence:
        # if (((eachsplit not in stoplist) and (len(eachsplit))) > 3 or eachsplit=='mom' or eachsplit=='dad'):
        if eachsplit not in stoplist:
            splitoutcome = splitoutcome+' '+eachsplit
    return splitoutcome.strip()


def processname(setence):   #name的命名模式
    int_findall = re.findall('\d{1}', setence)  #找含数字的字符串
    char_findall = re.findall('[\W]+',setence)  #找含下划线的字符串
    if char_findall:
        for eachone in char_findall:
            eachstring = setence.replace(eachone, '')
            setence = eachstring
    if int_findall:
        for eachone in int_findall:
            eachstring = setence.replace(eachone, ' ')
            setence = eachstring
    h = 1
    m = 0
    while(h):
        for each in keyword_gov_list:
            if setence.find(each) != -1:
                m=12
                h=0
                break
        if h==0: continue
        for each in keyword_media_list:
            if setence.find(each) != -1:
                m=13
                h=0
                break
        if h==0: continue
        J_findall = re.findall('[0-9]+.+[0-9]', setence)
        if J_findall:
            m = 13;
            h = 0;
            continue
        if setence.endswith('com'): m=11; h=0; continue
        A_findall = re.findall('[A-Z][a-z]+[A-Z][a-z]+[A-Z][a-z]+[A-Z][a-z]+', setence)
        if A_findall: m= 1 ; h=0;continue
        B_findall = re.findall('[A-Z][a-z]+[A-Z][a-z]+[A-Z][a-z]+', setence)
        if B_findall:m = 2;h = 0;continue
        C_findall = re.findall('[A-Z][a-z]+[A-Z][a-z]+[A-Z]+', setence)
        if C_findall:m = 3;h = 0;continue
        D_findall = re.findall('[A-Z]+[a-z]+[A-Z]+[a-z]+', setence)
        if D_findall:m = 4;h = 0;continue
        E_findall = re.findall('[A-Z]+[a-z]+[A-Z]+', setence)
        if E_findall:m = 5;h = 0;continue
        F_findall = re.findall('[A-Z]+[a-z]+', setence)
        if F_findall: m = 6;h = 0;continue
        G_findall = re.findall('[a-z]+[A-Z]+[a-z]*', setence)
        if G_findall: m = 7;h = 0;continue
        H_findall = re.findall('[A-Z]+', setence)
        if H_findall: m = 8;h = 0;continue
        I_findall = re.findall('[a-z]+', setence)
        if I_findall: m =9;h = 0;continue
        else:h=0
    return m

def screenname(setence):  #screenname的命名模式
    h = 1
    m = 0
    setence = setence.replace(" ", "")
    while(h):
        for each in keyword_gov_list:
            if setence.find(each) != -1:
                m=12
                h=0
                break
        if h==0: continue
        for each in keyword_media_list:
            if setence.find(each) != -1:
                m=13
                h=0
                break
        if h==0: continue
        J_findall = re.findall('[0-9]+.+[0-9]', setence)
        if J_findall:
            m = 13;
            h = 0;
            continue
        A_findall = re.findall('[A-Z]+[0-9]+', setence)
        if A_findall: m = 1; h = 0;continue
        B_findall = re.findall('[A-Z]+[a-z]+[0-9]+', setence)
        if B_findall: m = 2; h = 0;continue
        C_findall = re.findall('[a-z]+[0-9]+', setence)
        if C_findall: m = 3; h = 0;continue
        D_findall = re.findall('[A-Z]+[a-z]+[A-Z]+[a-z]+', setence)
        if D_findall: m = 4; h = 0;continue
        E_findall = re.findall('[A-Z]+[a-z]+', setence)
        if E_findall: m = 5; h = 0;continue
        F_findall = re.findall('[A-Z]+', setence)
        if F_findall: m = 6; h = 0;continue
        G_findall = re.findall('[a-z]+', setence)
        if G_findall: m = 7; h = 0;continue
        else:h=0
    return m

# 停用词表
stoplist = {}.fromkeys([line.strip() for line in
                        open(r'D:\PhD\research\机器学习用户分类\descriptionstopwords.txt','r',encoding='UTF-8')])

# government命名关键词
keyword_gov_list = {}.fromkeys([line.strip() for line in
                        open(r'C:\Users\Chensj\Desktop\keyword_gov.txt','r',encoding='UTF-8')])

# media命名关键词
keyword_media_list = {}.fromkeys([line.strip() for line in
                        open(r'C:\Users\Chensj\Desktop\keyword_media.txt','r',encoding='UTF-8')])

# 数据预处理
def DataProcessing(q,index):
    Process_id = 'Process-' + str(index)
    conn1 = pymysql.connect("localhost","root","baobei007","hurricane_data")
    cursor1 = conn1.cursor()
    while not q.empty():
        list=q.get()
        user_id = list[0]
        similarank = namesimilarize(list[1], list[2])
        name = processname(list[1])
        screen_name = screenname(list[2])
        FeatureStr = list[3]
        label = list[4]
        verified = list[5]
        followers = str(list[6])
        friends = str(list[7])
        listed = str(list[8])

        followers_category = Intrans(followers)
        friends_category = Intrans(friends)
        listed_category = Intrans(listed)
        FeatureStr = FeatureStr.lower()
        FeatureStr = deleteunusefulchar(FeatureStr)

        cursor1.execute("INSERT INTO users_normalized_three_classes_institution(user_id,name,screen_name,namesim,verified,listed_count,friends_count,followers_count,label,description) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",(user_id,name,screen_name,similarank,verified,listed_category,friends_category,followers_category,label,FeatureStr))
        print(Process_id, q.qsize())
    cursor1.close()
    conn1.commit()
    conn1.close()

if __name__=='__main__':#可以不用多进程
    start = time.time()
    allData = 29388
    dataOfEach = 29388
    batch = math.ceil(allData / dataOfEach)
    BATctrl = 0
    while BATctrl < batch:
        # 读数据
        conn2 = pymysql.connect("localhost","root","baobei007","hurricane_data")
        cursor2 = conn2.cursor()
        sql="select user_id,name,screen_name,description,label,verified,followers_count,friends_count,listed_count from users_labeled where label=1 limit "+str(dataOfEach * BATctrl)+","+str(dataOfEach)
        # sql = "select * from users_sample_three_classes_institution"
        cursor2.execute(sql)
        print('select begin:' + str(dataOfEach * BATctrl))
        results = cursor2.fetchall()
        results = list(results)
        BATctrl += 1
        manager = Manager()
        workQueue = manager.Queue(dataOfEach)
        # 数据预处理 DataProcessing
        for result in results:
            workQueue.put(result)
        pool =multiprocessing.Pool(processes=3)
        for i in range(3):
            pool.apply(DataProcessing, args=(workQueue,i))
        print("Started processes")
        pool.close()
        pool.join()
        cursor2.close()
        conn2.commit()
        conn2.close()
    end = time.time()
    print('Pool + Queue多进程爬虫的总时间为：', end - start)
    print('Main process Ended')


import json

index=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p']

# for indx in index:
#     f = open('alics_data_' + indx, encoding='utf-8', mode='r', errors='ignore')
#     w = open('alics_data_' + indx + '.json', encoding='utf-8', mode='w', errors='ignore')
#     text = f.readline().split('\001')
#     while text != ['']:
#         dicts = dict()
#         dicts['fold'] = 'validation'
#         dicts['topic'] = text[-1]
#         listDialogue = list()
#         # print(text)
#         t = json.loads('[' + str(text[2:-2][0]) + ']')
#
#         for j in t:
#             data = {
#                 'emotion': "no_emotion",
#                 'act': "no_act",
#                 'text': j['text'],
#                 'member_type': j['member_type'],
#                 'id': int(j['id'])
#             }
#             listDialogue.append(data)
#         listDialogue.sort(key=lambda x: x["id"])
#         dicts['dialogue'] = listDialogue
#         json_data = json.dumps(dicts, ensure_ascii=False)
#         w.write(json_data + '\n')
#         text = f.readline().split('\001')


# ----------------------------------------------------------------
# 合并数据集

# w = open('allData.json', encoding='utf-8', mode='w', errors='ignore')
# for indx in index:
#     f = open('alics_data_' + indx+'.json', encoding='utf-8', mode='r', errors='ignore')
#     tests=f.readlines()
#     w.writelines(tests)
#     f.close()

# ----------------------------------------------------------------
# 随机化并划分数据集

# from sklearn.utils import shuffle
# import random
# random.seed(3407)
#
# r = open('allData.json', encoding='utf-8', mode='r', errors='ignore')
# trainData=open('trainData.json', encoding='utf-8', mode='w', errors='ignore')
# valData=open('valData.json', encoding='utf-8', mode='w', errors='ignore')
# testData=open('testData.json', encoding='utf-8', mode='w', errors='ignore')
#
# texts=r.readlines()
# lenth=len(texts)
#
# texts = shuffle(texts)
#
# train_size = int(0.8 * lenth)
# val_size = int(0.1 * lenth)
# test_size = lenth - train_size - val_size
#
# train_data = texts[:train_size]
# val_data = texts[train_size:train_size+val_size]
# test_data = texts[train_size+val_size:]
#
# for i in train_data:
#     data=json.loads(i)
#     data['fold']='train'
#     d=json.dumps(data, ensure_ascii=False)
#     trainData.write(d+'\n')
#
# for i in test_data:
#     data=json.loads(i)
#     data['fold']='test'
#     d=json.dumps(data, ensure_ascii=False)
#     testData.write(d+'\n')
#
# valData.writelines(val_data)

# ----------------------------------------------------------------

import numpy as np
r = open('trainDataCM.json', encoding='utf-8', mode='r', errors='ignore')
w = open('trainDataCMDAG1.json', encoding='utf-8', mode='w', errors='ignore')
# listAllData=[]
w.write('[')
count=0
for i in r:
    listDialogue=list()
    data = json.loads(i)
    for j in data['dialogue']:
        d={
            'text': j['text'],
            "label": data['topic'],
            "speaker": j['member_type'],
            # "cls":np.random.rand(1024),
            # "cls":str(list(np.random.rand(1024))),
            "cls": []
        }
        # print(d)
        listDialogue.append(d)
    if count==0:
        count=count+1
        w.write(json.dumps(listDialogue, ensure_ascii=False)+'\n')
    else:
        w.write(','+json.dumps(listDialogue, ensure_ascii=False)+'\n')
    # listAllData.append(listDialogue)
w.write(']')


# ---------------------------------------------------------------------
# 标签信息
# r = open('allData.json', encoding='utf-8', mode='r', errors='ignore')
# speakerDic = dict()
# labelDic = dict()
# for i in r:
#     data = json.loads(i)
#     topic = data['topic']
#     labelDic.setdefault(topic, len(labelDic))
#     # labelDic[topic]=labelDic[topic]+1
#     for j in data['dialogue']:
#         member_type=j['member_type']
#         speakerDic.setdefault(member_type, len(speakerDic))
#         # speakerDic[member_type] = speakerDic[member_type] + 1
#
# print(labelDic)
# print(speakerDic)
# print(list(labelDic.keys()))
# print(list(speakerDic.keys()))
# count=0
# for k,v in labelDic.items():
#     # print(k,v)
#     count=count+v
# print(count)


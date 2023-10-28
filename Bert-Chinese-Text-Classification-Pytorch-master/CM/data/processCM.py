#
# f=open('callreason.train.fj_and_sh.2w','r',encoding='utf-8')
# texts=f.readlines()
# count=0
# dictCM=dict()
# for n,i in enumerate(texts):
#     t=i.split()
#     print(t)
#     if(len(t)==1):
#         t.extend(next(texts)[1].split())
#     if(len(t)>2):
#         count=count+1
#         dictCM.setdefault(t[2],0)
#         dictCM[t[2]]=dictCM[t[2]]+1
#
# dictCM=sorted(dictCM.items(),key=lambda x:x[1],reverse=True)
# print(count)
# print(dictCM)
# print(len(dictCM))

# -------------------------------------------------------------------------------
# import json
# f=open('callreason.train.fj_and_sh.2w','r',encoding='utf-8')
# w = open('CMdata.json', encoding='utf-8', mode='w', errors='ignore')
#
# count=0
# text=f.readline().split()
# dictCM=dict()
# while text :
#     dicts = dict()
#     if (len(text)==3):
#         count=count+1
#         dicts['fold'] = 'validation'
#         dicts['topic'] = text[2]
#         dictCM.setdefault(text[2], 0)
#         dictCM[text[2]] = dictCM[text[2]] + 1
#         listDialogue = list()
#         text=f.readline().split()
#         while(len(text)==2):
#             data = {
#                 'emotion': "no_emotion",
#                 'act': "no_act",
#                 'text': text[1],
#                 'member_type': text[0],
#             }
#             listDialogue.append(data)
#             text = f.readline().split()
#             while (len(text)==1):
#                 text = f.readline().split()
#         dicts['dialogue'] = listDialogue
#         json_data = json.dumps(dicts, ensure_ascii=False)
#         w.write(json_data + '\n')
#     text=f.readline().split()
#     if(len(text)==1):
#         text.extend(f.readline().split())
#         print(text)
#
# dictCM=sorted(dictCM.items(),key=lambda x:x[1],reverse=True)
# print(dictCM)
# print(len(dictCM))
# print(count)
# for i in dictCM:
#     print(i[0],i[1])

# -------------------------------------------------------------------
# 随机化并划分数据集
# from sklearn.utils import shuffle
# import random
# import json
# random.seed(3407)
#
# r = open('CMdata.json', encoding='utf-8', mode='r', errors='ignore')
# trainData=open('trainDataCM.json', encoding='utf-8', mode='w', errors='ignore')
# valData=open('valDataCM.json', encoding='utf-8', mode='w', errors='ignore')
# testData=open('testDataCM.json', encoding='utf-8', mode='w', errors='ignore')
#
# texts=r.readlines()
# lenth=len(texts)
# print(lenth)
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
# -----------------------------------------------------
# import json
#
# trainData=open('trainDataCM.json', encoding='utf-8', mode='r', errors='ignore')
# valData=open('valDataCM.json', encoding='utf-8', mode='r', errors='ignore')
# testData=open('testDataCM.json', encoding='utf-8', mode='r', errors='ignore')
#
# dictCMTrain=dict()
# dictCMval=dict()
# dictCMtest=dict()
# for i in trainData:
#     train=json.loads(i)
#     dictCMTrain.setdefault(train['topic'],0)
#     dictCMTrain[train['topic']] = dictCMTrain[train['topic']] + 1
#
# for j in valData:
#     val=json.loads(j)
#     dictCMval.setdefault(val['topic'],0)
#     dictCMval[val['topic']] = dictCMval[val['topic']] + 1
#
# for k in testData:
#     test=json.loads(k)
#     dictCMtest.setdefault(test['topic'],0)
#     dictCMtest[test['topic']] = dictCMtest[test['topic']] + 1
#
# dictCMTrain=sorted(dictCMTrain.items(),key=lambda x:x[1],reverse=True)
# dictCMval=sorted(dictCMval.items(),key=lambda x:x[1],reverse=True)
# dictCMtest=sorted(dictCMtest.items(),key=lambda x:x[1],reverse=True)
#
# for i in dictCMTrain:
#     print(i[0],i[1])
#
# print('----------------------------------------------------------')
#
# for i in dictCMval:
#     print(i[0],i[1])
#
#
# print('----------------------------------------------------------')
# for i in dictCMtest:
#     print(i[0],i[1])
# print('----------------------------------------------------------')
# print(len(dictCMTrain))
# print(len(dictCMval))
# print(len(dictCMtest))

# -----------------------------------------------------------
import json

trainData=open('trainDataCM.json', encoding='utf-8', mode='r', errors='ignore')
valData=open('valDataCM.json', encoding='utf-8', mode='r', errors='ignore')
testData=open('testDataCM.json', encoding='utf-8', mode='r', errors='ignore')

trainText=open('train.txt', encoding='utf-8', mode='w', errors='ignore')
valText=open('dev.txt', encoding='utf-8', mode='w', errors='ignore')
testText=open('test.txt', encoding='utf-8', mode='w', errors='ignore')

tagList=open('class.txt', encoding='utf-8', mode='r', errors='ignore')
tags=list()
for t in tagList:
    tags.append(t)

for i in trainData:
    train=json.loads(i)
    text=''
    for dlg in train['dialogue']:
        text=text+' '+dlg['text']
    trainText.write(text+'\t'+str(tags.index(train['topic']+'\n')))
    trainText.write('\n')

for j in valData:
    val=json.loads(j)
    text=''
    for dlg in val['dialogue']:
        text=text+' '+dlg['text']
    valText.write(text+'\t'+str(tags.index(val['topic']+'\n')))
    valText.write('\n')

for k in testData:
    test=json.loads(k)
    text=''
    for dlg in test['dialogue']:
        text=text+' '+dlg['text']
    testText.write(text+'\t'+str(tags.index(test['topic']+'\n')))
    testText.write('\n')

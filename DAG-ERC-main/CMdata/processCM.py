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

# ----------------------------------------------------------------------------------
# import json
#
# r = open('valDataCM.json', encoding='utf-8', mode='r', errors='ignore')
# w = open('valDataCMDAG1.json', encoding='utf-8', mode='w', errors='ignore')
# # listAllData=[]
# w.write('[')
# count=0
# for i in r:
#     listDialogue=list()
#     data = json.loads(i)
#     for j in data['dialogue']:
#         d={
#             'text': j['text'],
#             "label": data['topic'],
#             "speaker": j['member_type'],
#             # "cls":np.random.rand(1024),
#             # "cls":str(list(np.random.rand(1024))),
#             "cls": []
#         }
#         # print(d)
#         listDialogue.append(d)
#     if count==0:
#         count=count+1
#         w.write(json.dumps(listDialogue, ensure_ascii=False)+'\n')
#     else:
#         w.write(','+json.dumps(listDialogue, ensure_ascii=False)+'\n')
#     # listAllData.append(listDialogue)
# w.write(']')

# -----------------------------------------------------
import pickle
# labelV={
#     'stoi': {'恶意不退货仅退款\n': 0, '对未收货商品虚假评价\n': 1, '恶意发起规则投诉\n': 2, '广告评论\n': 3, '同行/买家因纠纷报复我\n': 4, '无意义评论\n': 5, '使用错误收货信息\n': 6, '骗取运费险\n': 7, '利用评价要挟\n': 8, '评论泄露隐私\n': 9, '辱骂侮辱的评论\n': 10, '退货空包少件调包\n': 11, '订单留言发布垃圾消息\n': 12, '虚假物流退货\n': 13},
#     'itos': ['恶意不退货仅退款\n', '对未收货商品虚假评价\n', '恶意发起规则投诉\n', '广告评论\n', '同行/买家因纠纷报复我\n', '无意义评论\n', '使用错误收货信息\n', '骗取运费险\n', '利用评价要挟\n', '评论泄露隐私\n', '辱骂侮辱的评论\n', '退货空包少件调包\n', '订单留言发布垃圾消息\n', '虚假物流退货\n']
# }
labelV={
    'stoi': {'业务使用问题': 0, '营销活动信息': 1, '办理方式': 2, '变更': 3, '账户信息': 4, '不知情定制问题': 5, '开通': 6, '业务规定': 7, '取消': 8, '费用问题': 9, '业务资费': 10, '业务订购信息查询': 11, '下载/设置': 12, '营销问题': 13, '重置/修改/补发': 14, '移机/装机/拆机': 15, '产品/业务功能': 16, '号码状态': 17, '用户资料': 18, '使用方式': 19, '服务问题': 20, '补换卡': 21, '信息安全问题': 22, '业务办理问题': 23, '停复机': 24, '打印/邮寄': 25, '销户/重开': 26, '服务渠道信息': 27, '工单处理结果': 28, '业务规定不满': 29, '宽带覆盖范围': 30, '网络问题': 31, '电商货品信息': 32, '缴费': 33, '无声电话': 34, '电商售后问题': 35, '骚扰电话': 36},
    'itos': ['业务使用问题', '营销活动信息', '办理方式', '变更', '账户信息', '不知情定制问题', '开通', '业务规定', '取消', '费用问题', '业务资费', '业务订购信息查询', '下载/设置', '营销问题', '重置/修改/补发', '移机/装机/拆机', '产品/业务功能', '号码状态', '用户资料', '使用方式', '服务问题', '补换卡', '信息安全问题', '业务办理问题', '停复机', '打印/邮寄', '销户/重开', '服务渠道信息', '工单处理结果', '业务规定不满', '宽带覆盖范围', '网络问题', '电商货品信息', '缴费', '无声电话', '电商售后问题', '骚扰电话']}

#
spreakerV={
    'stoi': {'1': 0, '2': 1,'3': 2},
    'itos': ['1', '2', '3']
}
#
with open('labelFile.pkl', 'wb') as file:
    pickle.dump(labelV, file)

with open('speakerFile.pkl', 'wb') as file:
    pickle.dump(spreakerV, file)

# pickle.dump(labelV,labelFile)
# pickle.dump(spreakerV, speakerFile)

label_vocab = pickle.load(open('labelFile.pkl', 'rb'))
print(label_vocab)

speaker_vocab = pickle.load(open('speakerFile.pkl', 'rb'))
print(speaker_vocab)

# ---------------------------------------------------
# import json
# trainData=open('CMdata.json', encoding='utf-8', mode='r', errors='ignore')
# dictCMTrain=dict()
# for i in trainData:
#     train=json.loads(i)
#     dictCMTrain.setdefault(train['topic'], 0)
#     dictCMTrain[train['topic']] = dictCMTrain[train['topic']] + 1
#
# print(dictCMTrain)
# count=0
# labelV={
#      'stoi': {},
#      'itos': []
# }
# list1=[]
# for k,v in dictCMTrain.items():
#     print(k,v)
#     dictCMTrain[k]=count
#     count=count+1
#     list1.append(k)
#
# print(dictCMTrain)
# print(list1)
#
# labelV['stoi']=dictCMTrain
# labelV['itos']=list1
# print(labelV)
#

import json
import csv
r = open('testDataCM.json', encoding='utf-8', mode='r', errors='ignore')
train = open('trainDataCM.json', encoding='utf-8', mode='r', errors='ignore')
val= open('valDataCM.json', encoding='utf-8', mode='r', errors='ignore')

testData=open('CM.csv',encoding='utf-8',mode='w')
header = ['text','label','train']
writer = csv.writer(testData)
writer.writerow(header)

for i in train:
    text=json.loads(i)
    data = list()
    t=''
    for j in text['dialogue']:
        t=t+' '+j['text']
    data.append(t)
    data.append(text['topic'])
    data.append(text['fold'])
    writer.writerow(data)
    data.clear()



# for i in val:
#     text=json.loads(i)
#     data = list()
#     t=''
#     for j in text['dialogue']:
#         t=t+' '+j['text']
#     data.append(t)
#     data.append(text['topic'])
#     data.append(text['fold'])
#     writer.writerow(data)
#     data.clear()

for i in r:
    text=json.loads(i)
    data = list()
    t=''
    for j in text['dialogue']:
        t=t+' '+j['text']
    data.append(t)
    data.append(text['topic'])
    data.append(text['fold'])
    writer.writerow(data)
    data.clear()


r.close()








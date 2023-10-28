import json

trainData=open('trainData.json', encoding='utf-8', mode='r', errors='ignore')
valData=open('valData.json', encoding='utf-8', mode='r', errors='ignore')
testData=open('testData.json', encoding='utf-8', mode='r', errors='ignore')

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
        text=text+' '+dlg['text'].replace('\n',' ').replace('\r',' ').replace('\t',' ')
    trainText.write(text+'\t'+str(tags.index(train['topic'])))
    trainText.write('\n')

for j in valData:
    val=json.loads(j)
    text=''
    for dlg in val['dialogue']:
        text=text+' '+dlg['text'].replace('\n',' ').replace('\r',' ').replace('\t',' ')
    valText.write(text+'\t'+str(tags.index(val['topic'])))
    valText.write('\n')

for k in testData:
    test=json.loads(k)
    text=''
    for dlg in test['dialogue']:
        text=text+' '+dlg['text'].replace('\n',' ').replace('\r',' ').replace('\t',' ')
    print(text)
    print('xx')
    testText.write(text+'\t'+str(tags.index(test['topic'])))
    testText.write('\n')

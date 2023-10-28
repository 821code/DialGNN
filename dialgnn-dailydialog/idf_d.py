import nltk, json, pandas as pd, numpy as np, pickle
from transformers import BertModel,BertTokenizer
import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, BertModel
import math
import re

BERT_PATH = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertModel.from_pretrained(BERT_PATH)
special_tokens = [tokenizer.unk_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.mask_token_id]
pad_embedding = model(torch.tensor([tokenizer.pad_token_id]).unsqueeze(0))
pad_embedding = pad_embedding[0].squeeze(0).squeeze(0) 

labels_to_idx = {'ordinary_life':0, 'school_life':1, 'culture_and_educastion':2, 'attitude_and_emotion':3, 'relationship':4, 'tourism':5 , 'health':6, 'work':7, 'politics':8, 'finance':9}
idx_to_labels = dict()
for key in labels_to_idx:
    idx_to_labels[labels_to_idx[key]] = key

idf_d = dict()

def get_dialogue_graph(filename):
    with open(filename + '.json', 'r') as f:
        for line in f:
            s = eval(line)
            labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            labels[labels_to_idx[s['topic']]] = 1
            dialogue = []
            for item in s['dialogue']:
	            dialogue.append(item['text'])

            dialogue_bert_idx = [] 
            dialogue_bert_idx.insert(0, tokenizer.cls_token_id)
            for input_text in dialogue:
                input_ids = tokenizer.encode(input_text, add_special_tokens=True) 
                input_ids.pop(0), input_ids.pop()
                input_ids.insert(0, tokenizer.sep_token_id)
                dialogue_bert_idx.extend(input_ids)
            dialogue_bert_idx.append(tokenizer.sep_token_id)

            dialogue_bert_output = [] 
            dialogue_bert_idx_tmp = dialogue_bert_idx
            while len(dialogue_bert_idx_tmp) > 512:
                dialogue_bert_idx_trunc = dialogue_bert_idx_tmp[:512]
                dialogue_bert_idx_tmp = dialogue_bert_idx_tmp[512:]
                dialogue_bert_output_tmp = model(torch.tensor(dialogue_bert_idx_trunc).unsqueeze(0))
                dialogue_bert_output.extend(dialogue_bert_output_tmp[0].squeeze(0))
            dialogue_bert_output_tmp = model(torch.tensor(dialogue_bert_idx_tmp).unsqueeze(0))
            dialogue_bert_output.extend(dialogue_bert_output_tmp[0].squeeze(0))

            idx_cnt = dict()
            for idx in dialogue_bert_idx:
                if idx in idx_cnt.keys():
                    idx_cnt[idx] = idx_cnt[idx] + 1
                else:
                    idx_cnt[idx] = 1
            nodes_idx = list(idx_cnt.keys()) 

            for key in nodes_idx:
                if key in idf_d.keys():
                    idf_d[key] += 1
                else:
                    idf_d[key] = 1
                            
            graph_info = 1
            yield graph_info

for filename in ['train', 'test']:
	dialogue_graph = get_dialogue_graph(filename)
	if filename == 'train':
		for i in range(11118):
		    x = next(dialogue_graph)
	else:
		for i in range(1000):
		    x = next(dialogue_graph)

	f=open('idf_d_' + filename + '.txt','wb')  
	pickle.dump(idf_d,f,0)  
	f.close()  
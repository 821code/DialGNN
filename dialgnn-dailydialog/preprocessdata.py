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
pad_embedding = pad_embedding[0].squeeze(0).squeeze(0) #pad的bert embedding

labels_to_idx = {'ordinary_life':0, 'school_life':1, 'culture_and_educastion':2, 'attitude_and_emotion':3, 'relationship':4, 'tourism':5 , 'health':6, 'work':7, 'politics':8, 'finance':9}
idx_to_labels = dict()
for key in labels_to_idx:
    idx_to_labels[labels_to_idx[key]] = key
    

def get_dialogue_graph(filename, idf_D):
    global maxedge, maxnode
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
            nodes_idx.sort()

            tf_N = len(dialogue_bert_idx)
            for sep in special_tokens:  
                if sep in nodes_idx:
                    tf_N -= idx_cnt[sep]

            for sepcial_token in special_tokens:
                if sepcial_token not in nodes_idx:
                    continue
                pos = nodes_idx.index(sepcial_token)
                if sepcial_token == tokenizer.sep_token_id:
                    for _ in range(idx_cnt[sepcial_token] - 2):
                        nodes_idx.insert(pos, sepcial_token)
                else:
                    for _ in range(idx_cnt[sepcial_token] - 1):
                        nodes_idx.insert(pos, sepcial_token)

            nodes_embedding = list()
            nodes_embedding_dict = dict()
            sepcnt = 0
            for i, idx in enumerate(dialogue_bert_idx):
                if idx in special_tokens:
                    if idx == tokenizer.sep_token_id:
                        sepcnt = sepcnt + 1
                        if sepcnt < idx_cnt[tokenizer.sep_token_id]:
                            nodes_embedding.append(dialogue_bert_output[i]) 
                    else:
                        nodes_embedding.append(dialogue_bert_output[i])
                elif idx in nodes_embedding_dict.keys():
                    nodes_embedding_dict[idx] = nodes_embedding_dict[idx] + dialogue_bert_output[i]
                else:
                    nodes_embedding_dict[idx] = torch.zeros(768)
            for idx in nodes_idx:
                if idx not in special_tokens:  
                    nodes_embedding.append(nodes_embedding_dict[idx] / idx_cnt[idx])

            nodes_embedding_numpy = []
            for i in nodes_embedding:
                nodes_embedding_numpy.append(i.detach().numpy())

            if len(nodes_embedding_numpy) > maxnode:
                maxnode = len(nodes_embedding_numpy)
            while len(nodes_embedding_numpy) < 350: 
                nodes_embedding_numpy.append(pad_embedding.detach().numpy())
            edges_list = [] 
            edge_weights_list  = []
            pos_cls = nodes_idx.index(tokenizer.cls_token_id)
            pos_sep = pos_cls
            for value in dialogue_bert_idx:
                if value == tokenizer.sep_token_id:
                    pos_sep = pos_sep + 1
                    continue
                elif value in special_tokens:
                    continue
                tf_idf = idx_cnt[value] / tf_N * math.log10(idf_D / (1 + idf_d[value]))
                edges_list.append([pos_sep, nodes_idx.index(value)])
                edges_list.append([pos_cls, nodes_idx.index(value)])
                edge_weights_list.extend([tf_idf, tf_idf])

            if len(edges_list) > maxedge:
                maxedge = len(edges_list)
            while len(edges_list) < 1782:
                edges_list.append([348, 348])
                edge_weights_list.extend([0])
            edges = np.array(edges_list).T
            edge_weights = tf.cast(
                np.array(edge_weights_list), dtype=tf.dtypes.float32
            )
            node_features = tf.cast(
                np.array(nodes_embedding_numpy), dtype=tf.dtypes.float32
            )
            graph_info = (node_features, edges, edge_weights, labels, pos_cls)
            yield graph_info
        
for filename in ['test',  'train']:
	f=open('idf_d_' + filename + 'txt','rb')  
	idf_d=pickle.load(f)  
	f.close()
	idf_D = 1000
	if filename == 'train':
		idf_D = 11118		
	dialogue_graph = get_dialogue_graph(filename, idf_D)
	node_features = list()
	edges = list()
	edge_weights = list()
	x_train = list()
	y_train = list()
	pos_cls = list()
	labels = list()
	pos_cls = list()
	
	for i in range(idf_D): 
	    node_features_tmp, edges_tmp, edge_weights_tmp, labels_tmp, pos_cls_tmp, mask_tmp = next(dialogue_graph)
	    node_features.append(list(node_features_tmp))
	    edges.append(list(edges_tmp))
	    edge_weights.append(list(edge_weights_tmp))
	    labels.append(labels_tmp)
	    pos_cls.append(pos_cls_tmp)
	    

	node_features = np.array(node_features)  #每行长度要一样，，好像要先padding
	edges = np.array(edges)
	edge_weights = np.array(edge_weights)
	labels = np.array(labels)
	pos_cls = np.array(pos_cls)
	graph_info = node_features, edges, edge_weights, labels, pos_cls
	
	f=open(filename + '.txt','wb')  
	pickle.dump(graph_info,f,0)  
	f.close()  
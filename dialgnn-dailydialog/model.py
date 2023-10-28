import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import layers
from transformers import BertTokenizer, BertModel
import pickle
import math
import warnings
from sklearn.metrics import f1_score
import keras.backend as K
from sklearn.metrics import average_precision_score
from keras.utils import to_categorical

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 6)
pd.set_option("display.max_rows", 6)
np.random.seed(2)

labels_to_idx = {'ordinary_life':0, 'school_life':1, 'culture_and_educastion':2, 'attitude_and_emotion':3, 'relationship':4, 'tourism':5 , 'health':6, 'work':7, 'politics':8, 'finance':9}
idx_to_labels = dict()
for key in labels_to_idx:
    idx_to_labels[labels_to_idx[key]] = key


class GraphAttention(layers.Layer):
    def __init__(
        self,
        units,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)

    def build(self, input_shape):

        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel",
        )
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            name="kernel_attention",
        )
        self.built = True

    def call(self, inputs):
        node_states, edges, edge_weights = inputs
        node_states, edges, edge_weights = node_states[0], edges[0], edge_weights[0]
        node_states_transformed = tf.matmul(node_states, self.kernel)
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(
            node_states_expanded, (tf.shape(edges)[0], -1)
        )
        node_states_expanded = node_states_expanded * tf.expand_dims(edge_weights, -1)
        attention_scores = tf.nn.leaky_relu(
            tf.matmul(node_states_expanded, self.kernel_attention)
        )
        attention_scores = tf.squeeze(attention_scores, -1)
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2))
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments = tf.shape(node_states)[0]
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )      
        attention_scores_norm = attention_scores / attention_scores_sum
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out


class MultiHeadGraphAttention(layers.Layer):
    def __init__(self, units, num_heads=8, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttention(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices, edge_weights = inputs
        outputs = [
            attention_layer([atom_features, pair_indices, edge_weights])
            for attention_layer in self.attention_layers
        ]
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        return tf.nn.relu(outputs)

class GraphAttentionNetwork(keras.Model):
    def __init__(
        self,
        node_states,
        edges,
        edge_weights, 
        pos_cls,
        hidden_units,
        num_heads,
        num_layers,
        output_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.edge_weights = edge_weights
        self.pos_cls = pos_cls
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttention(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges, edge_weights, pos_cls = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges, edge_weights]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data
        self.node_states, self.edges, self.edge_weights, self.pos_cls = indices
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights) 
        labels = labels[0]
        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.edges, self.edge_weights, self.pos_cls])
            outputs = outputs[0]
            loss = self.compiled_loss(labels, tf.gather(outputs, self.pos_cls[0]))
        grads = tape.gradient(loss, self.trainable_weights)
        optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, self.pos_cls[0]))
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        self.node_states, self.edges, self.edge_weights, self.pos_cls = indices
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights) 
        labels = labels[0]
        outputs = self([self.node_states, self.edges, self.edge_weights, self.pos_cls])
        outputs = outputs[0]
        return tf.nn.softmax(tf.gather(outputs, self.pos_cls[0]))

    def test_step(self, data):
        indices, labels = data
        self.node_states, self.edges, self.edge_weights, self.pos_cls = indices
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights) 
        labels = labels[0]
        outputs = self([self.node_states, self.edges, self.edge_weights, self.pos_cls])
        outputs = outputs[0]
        loss = self.compiled_loss(labels, tf.gather(outputs, self.pos_cls[0]))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, self.pos_cls[0]))
        return {m.name: m.result() for m in self.metrics}

HIDDEN_UNITS = 50
NUM_HEADS = 4
NUM_LAYERS = 3
OUTPUT_DIM = 10

NUM_EPOCHS = 100
BATCH_SIZE = 4
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 3e-4
MOMENTUM = 0.9

loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
accuracy_fn = keras.metrics.CategoricalAccuracy(name="acc")
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-5, patience=5, restore_best_weights=True)

f=open('train.txt','rb')  
graph=pickle.load(f)
f.close()
node_states, edges, edge_weights, labels, pos_cls = graph[0], graph[1], graph[2], graph[3], graph[4]
edges = [edge.T for edge in edges]
edge_weights = [weight.T for weight in edge_weights]
node_states = tf.convert_to_tensor(node_states)
edges = tf.convert_to_tensor(edges)
edge_weights = tf.convert_to_tensor(edge_weights)
labels = np.array(labels)

gat_model = GraphAttentionNetwork(
    node_states, edges, edge_weights, pos_cls, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
)

gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

train_labels = labels
train_indices =  (node_states, edges, edge_weights, pos_cls)

gat_model.fit(
    x=train_indices,
    y=train_labels,
    validation_split=VALIDATION_SPLIT,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping],
    verbose=2
)

f=open('test.txt','rb')  
graph=pickle.load(f)
f.close()

node_states, edges, edge_weights, labels, pos_cls = graph[0], graph[1], graph[2], graph[3], graph[4]
edges = [edge.T for edge in edges]
edge_weights = [weight.T for weight in edge_weights]
node_states = tf.convert_to_tensor(node_states)
edges = tf.convert_to_tensor(edges)
edge_weights = tf.convert_to_tensor(edge_weights)
labels = np.array(labels)

test_indices =  (node_states, edges, edge_weights, pos_cls)

_, test_accuracy = gat_model.evaluate(x=test_indices, y=labels, verbose=2)

print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")

y_pred = gat_model(train_indices, training = False)
y_pred = tf.nn.softmax(y_pred)
yy_pred = []
for i in range(labels.shape[0]):
	y_max_positions = np.argmax(y_pred[i][pos_cls[i]], axis = 0)
	y_multilabel = to_categorical(y_max_positions, num_classes=10)
	yy_pred.append(y_multilabel)
yy_pred = np.array(yy_pred)

print("F1 macro : ", f1_score(labels, yy_pred, average='macro'))
print("F1 weighted : ", f1_score(labels, yy_pred, average='weighted'))
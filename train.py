from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import matplotlib
matplotlib.use('Agg')


dataset = 'data'

# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size, x_shape0, labels_in = load_corpus(
    FLAGS.dataset)

# print("features_type:", type(features))
#
# # print("support_type:", type(support))
# print("y_train_type:", type(y_train))
# print("train_mask_type:", type(train_mask))

# DEBUG_NUM = 30000
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
#     adj[:DEBUG_NUM, :DEBUG_NUM],features[:DEBUG_NUM, :],y_train[:DEBUG_NUM, :],\
#     y_val[:DEBUG_NUM, :],y_test[:DEBUG_NUM, :],train_mask[:DEBUG_NUM],val_mask[:DEBUG_NUM],\
#     test_mask[:DEBUG_NUM]
print("train:features:")
print(adj)

features = sp.identity(features.shape[0])  # featureless
# features = features[:, :-300]

print("dadj:",adj.shape)
print("shape_feature", features.shape)

print("OK1")
# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
print("OK2")
# print("len of 0", len(features.nonzero()[0]))
# Define placeholders
placeholders = {
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)
}
print("OK3")
# Create model
print(features[2][1])
model = model_func(placeholders, input_dim=features[2][1], logging=True)
print("OK4")
# Initialize session
session_conf = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
print("OK5")
sess = tf.compat.v1.Session(config=session_conf)
print("OK6")

# def BFS(x,A):
#     ans, queue = set(),[x]
#     while queue:
#         current = queue.pop()
#         for i in range(A[current].shape[1]):
#             print(i)
#             if A[current,i] != 0 and i not in ans:
#                 ans.add(i)
#                 queue.append(i)
#     return ans
#
# subadj = []
# allnodes = set(range(adj.shape[0]))
# print("type of adj:", type(adj))
# print("adj:", adj.toarray())
# print("the first len of adj:", adj[0])
#
# for i in allnodes:
#     cache = BFS(i, adj)
#     subadj.append(cache)
#     allnodes = allnodes - cache
#
# for item in subadj:
#     print("len of subadj:", len(item))
# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    # 启动一个图，feed_dict为graph的输入值。
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)
print("OK7")

# Init variables
sess.run(tf.compat.v1.global_variables_initializer())

print("参数初始化完毕")
cost_val = []

# Train model
epoch = 0
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    # print("features_type:",type(features))

    # print("support_type:", type(support))
    # print("y_train_type:", type(y_train))
    # print("train_mask_type:", type(train_mask))
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # print("feature shapr:", features[2][1], y_train[2][1])

    # print("feed_dict更新完毕")

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.layers[0].embedding], feed_dict=feed_dict)
    # print("开始运行")
    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    # print("验证集掩码 and 测试集掩码：", val_mask, train_mask)
    # print("验证集acc and cost：", acc, cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print("test pred", len(test_pred))
print("test labels:", len(test_labels))
print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

# doc and word embeddings
print('embeddings:')
print("tha shape of word_embedding:", outs[3] .shape)
print("tha type of word_embedding:", type(outs[3]))
word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
train_doc_embeddings = outs[3][:train_size]  # include val docs
test_doc_embeddings = outs[3][adj.shape[0] - test_size:]
print("type of labels:", type(labels))
print("labels", labels)
y1 = labels_in[:train_size]
y2 = labels_in[adj.shape[0] - test_size:]
y = np.concatenate((y1, y2), axis=0)
print(len(word_embeddings), len(train_doc_embeddings),
      len(test_doc_embeddings))
print(word_embeddings)

f = open('vocab_data/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data_result/' + dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

train_doc_vectors = []
test_doc_vectors = []
test_doc_vectors_lstm = []
doc_vectors_k_means = []
doc_vectors_k_means_item = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vectors_k_means.append(doc_vector)
    train_doc_vector_str = ' '.join([str(x) for x in doc_vector])
    train_doc_vectors.append('doc_' + str(doc_id) + ' ' + train_doc_vector_str)
    doc_id += 1

doc_id = 0
for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    test_doc_vectors_lstm.append(doc_vector)
    # doc_vectors_k_means.append(doc_vector)
    test_doc_vector_str = ' '.join([str(x) for x in doc_vector])
    test_doc_vectors.append('doc_' + str(doc_id) + ' ' + test_doc_vector_str)
    doc_id += 1

# doc_embeddings_str = '\n'.join(doc_vectors)
# f = open('data_result/' + dataset + '_doc_vectors.txt', 'w')
# f.write(doc_embeddings_str)
# f.close()

train_doc_embeddings_str = '\n'.join(train_doc_vectors)
f = open('data_result/train_doc_vectors.txt', 'w')
f.write(train_doc_embeddings_str)
f.close()

doc_embeddings_str = '\n'.join(test_doc_vectors)
f = open('data_result/test_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()

print("type of y:", type(y))
print("y:", y)
print("所有文档向量长度以及标签长度：", len(test_doc_vectors), y.shape)

with open('data_result/doc_vectors.pk', 'wb+') as f:
    joblib.dump(doc_vectors_k_means, f)
with open('data_result/word_embedding.pk', 'wb+')as f:
    joblib.dump(word_embeddings, f)

with open('data_result/y.pk', 'wb+') as f:
    joblib.dump(y, f)

# print("doc_vectors_k_means:", doc_vectors_k_means)

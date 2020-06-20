import pickle as pkl
import numpy as np
import scipy.sparse as sp
from math import log
from sklearn.externals import joblib
import random
from tqdm import tqdm

doc_name_list = []
doc_train_list = []
doc_test_list = []
train_ids = []
test_ids = []
dataset = 'data'

real_train_doc_word = joblib.load('vocab_data/train_shuffle_doc_words_list.pk')
real_train_doc_label = joblib.load('vocab_data/train_shuffle_doc_label_list.pk')
real_train_doc_label_str = '\n'.join(real_train_doc_label)

real_test_doc_label = joblib.load('vocab_data/test_shuffle_doc_label_list.pk')
real_test_doc_word = joblib.load('vocab_data/test_shuffle_doc_words_list.pk')
real_test_doc_label_str = '\n'.join(real_test_doc_label)

train_size = len(real_train_doc_word)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size
test_size = len(real_test_doc_word)

print("test_szielllll:", test_size)
print("train_szielllll:", train_size)

f = open('graph_data/data.real_train.name', 'w')    #不包括验证集
f.write(real_train_doc_label_str)
f.close()

train_ids = list(range(train_size))
# random.shuffle(train_ids)
print("train_ids:", train_ids)
print("train_ids_len", len(train_ids))
print("ttttpye:", type(train_ids))

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('graph_data/data.train.index', 'w')          #包括验证集
f.write(train_ids_str)
f.close()

test_ids = list(range(test_size))
# random.shuffle(test_ids)

print("test_ids:", test_ids)
print("test_ids_len", len(test_ids))

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('graph_data/data.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
print(ids)
print(len(ids))

word_embeddings_dim = 100
word_vector_map = {}

word_id_map = joblib.load('vocab_data/word_id_map.pk')
word_doc_freq = joblib.load('vocab_data/word_doc_freq.pk')
vocab = joblib.load('vocab_data/vocab.pk')

vocab_size = len(vocab)

shuffle_doc_label_list = real_train_doc_label+real_test_doc_label
shuffle_doc_words_list = real_train_doc_word + real_test_doc_word
'''
Word definitions end
'''
# label list
# 存储所有标签，不重复。存在label_set集合里，然后转成label_list列表里。
label_set = set()
for doc_meta in shuffle_doc_label_list:
    label_set.add(doc_meta)
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('graph_data/data_labels.txt', 'w')
f.write(label_list_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in tqdm(range(real_train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

# 给训练数据的标签编码，编码成one-hot形式。[0，0，0，0，0....1，0，0]，一个向量，只有一个1.
# 存在y里，y是个列表，最后转换成np.array。
y = []
for i in tqdm(range(real_train_size)):
    label = shuffle_doc_label_list[i]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = sp.coo_matrix(y)
print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)
print("aaaaafter test_size:", test_size)
row_tx = []
col_tx = []
data_tx = []

for i in tqdm(range(test_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    # print("I + TRAIN SIZELLLLL:", i)
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in tqdm(range(test_size)):
    label = shuffle_doc_label_list[i + train_size]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = sp.coo_matrix(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words
# allx最后没有用，表示特征，没有用。

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in tqdm(range(len(vocab))):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in tqdm(range(train_size)):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in tqdm(range(vocab_size)):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in tqdm(range(train_size)):
    label = shuffle_doc_label_list[i]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in tqdm(range(vocab_size)):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)
# 单词的标签是00000000000

# ally = np.array(ally)
ally = sp.coo_matrix(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for d in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[d]
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

# 计算有多少个窗口包含有词XXX。
# 存在字典里。
word_window_freq = {}
for w in tqdm(range(len(windows))):
    window = windows[w]
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

# 计算两个词的共现次数。
# 存在字典里。
word_pair_count = {}
for w in tqdm(range(len(windows))):
    window = windows[w]
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

# 两个词之间的边的权重。
for key in tqdm(word_pair_count):
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

# doc word frequency
doc_word_freq = {}

for doc_id in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1
m = 0
for i in tqdm(range(len(shuffle_doc_words_list))):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        #     验证集。
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)
# print("the len of key:", m)
node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
print("Dumping...")

f = open("graph_data/ind.{}.x".format(dataset), 'wb')
tqdm(pkl.dump(x, f))
f.close()
print("xxxxxxxxxxxxx ok")

f = open("graph_data/ind.{}.y".format(dataset), 'wb')
tqdm(pkl.dump(y, f))
f.close()
print("yyyyyyyyyyyyyyyy ok")

f = open("graph_data/ind.{}.tx".format(dataset), 'wb')
tqdm(pkl.dump(tx, f))
f.close()
print("tttttttttttttxxxxxxxxxxxx ok")

f = open("graph_data/ind.{}.ty".format(dataset), 'wb')
tqdm(pkl.dump(ty, f))
f.close()
print("tttttttttttyyyyyyyyyyyy ok")

f = open("graph_data/ind.{}.allx".format(dataset), 'wb')
tqdm(pkl.dump(allx, f))
f.close()
print("aaaaaalllllllllllxxxxxxxxxxxxx ok")

f = open("graph_data/ind.{}.ally".format(dataset), 'wb')
tqdm(pkl.dump(ally, f))
f.close()
print("alllllllllyyyyyyyyyyyy ok")

f = open("graph_data/ind.{}.adj".format(dataset), 'wb')
tqdm(pkl.dump(adj, f))
f.close()
print("allllllllladj ok")

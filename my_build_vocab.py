
import random
from sklearn.externals import joblib

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
train_doc_label_list = []
test_doc_label_list = []

f = open('data/train_label.txt', 'r')
lines = f.readlines()
for line in lines:
    train_doc_label_list.append(line.strip('\n'))
    # 将标签存进doc_label_list里。是一个列表。
    # temp = line.split("\t")
    # if temp[1].find('test') != -1:
    # doc_list.append(line.strip())
    # elif temp[1].find('train') != -1:
    #     doc_train_list.append(line.strip())
f.close()
# print(train_doc_label_list)
# print(type(train_doc_label_list))

with open('data/test_label.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        test_doc_label_list.append(line.strip('\n'))


train_doc_content_list = []
test_doc_content_list = []
f = open('data/train_data_clean.txt', 'r')
lines = f.readlines()
for line in lines:
    train_doc_content_list.append(line.strip())
f.close()

with open('data/test_data_clean.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        test_doc_content_list.append(line.strip())
# print(doc_content_list)

train_doc_ids = []
test_doc_ids = []
for i in range(len(train_doc_label_list)):
    # train_id = doc_labe l_list.index(train_name)
    train_doc_ids.append(i)

for i in range(len(test_doc_label_list)):
    # train_id = doc_labe l_list.index(train_name)
    test_doc_ids.append(i)

random.shuffle(train_doc_ids)
random.shuffle(test_doc_ids)
# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]
# 打乱文档，并且记录下索引0、1、2、3、4、5、6......n。
train_doc_ids_str = '\n'.join(str(index) for index in train_doc_ids)
f = open('vocab_data/train_data.doc.index', 'w')
f.write(train_doc_ids_str)
f.close()

test_doc_ids_str = '\n'.join(str(index) for index in test_doc_ids)
f = open('vocab_data/test_data.doc.index', 'w')
f.write(test_doc_ids_str)
f.close()

train_shuffle_doc_label_list = []
train_shuffle_doc_words_list = []
test_shuffle_doc_label_list = []
test_shuffle_doc_words_list = []

for id in train_doc_ids:
    # 此处的doc_ids是打乱的。
    train_shuffle_doc_label_list.append(train_doc_label_list[int(id)])
    train_shuffle_doc_words_list.append(train_doc_content_list[int(id)])
train_shuffle_doc_label_str = '\n'.join(train_shuffle_doc_label_list)
train_shuffle_doc_words_str = '\n'.join(train_shuffle_doc_words_list)
print("the len of doc:", len(train_shuffle_doc_words_list))
print("the len of label:", len(train_shuffle_doc_label_list))

# 把标签打乱，存在data/CNESC_shuffle.txt文件里。
f = open('vocab_data/train_label_shuffle.txt', 'w')
f.write(train_shuffle_doc_label_str)
f.close()

# 把数据打乱，存在data/corpus/CNESC_shuffle.txt文件里。
f = open('vocab_data/train_data_shuffle.txt', 'w')
f.write(train_shuffle_doc_words_str)
f.close()

for id in test_doc_ids:
    # 此处的doc_ids是打乱的。
    test_shuffle_doc_label_list.append(test_doc_label_list[int(id)])
    test_shuffle_doc_words_list.append(test_doc_content_list[int(id)])
test_shuffle_doc_label_str = '\n'.join(test_shuffle_doc_label_list)
test_shuffle_doc_words_str = '\n'.join(test_shuffle_doc_words_list)
print("the len of doc:", len(test_shuffle_doc_words_list))
print("the len of label:", len(test_shuffle_doc_label_list))

# 把标签打乱，存在data/CNESC_shuffle.txt文件里。
f = open('vocab_data/test_label_shuffle.txt', 'w')
f.write(test_shuffle_doc_label_str)
f.close()

# 把数据打乱，存在data/corpus/CNESC_shuffle.txt文件里。
f = open('vocab_data/test_data_shuffle.txt', 'w')
f.write(test_shuffle_doc_words_str)
f.close()
'''''''''''''''''
'''''''''''''''''
shuffle_doc_words_list = train_shuffle_doc_words_list + test_shuffle_doc_words_list
# build vocab
word_freq = {}
''''
word_freq={'word':n}计算word的出现次数，存储在word_freq里
'''''
word_set = set()
'''''''''
word_set为集合，存储所有的word，不重复，最后转换为列表赋值给vocab。
'''''''''
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        # 集合word_set，里面不会有重复的元素。
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1


vocab = list(word_set)
vocab_size = len(vocab)
print("the len of vocab:", len(vocab))

word_doc_list = {}
'''''''''
word_doc_list={'word':[2,6,10,9,89,100,24354,......]}
找到一个word在哪篇文档中出现。
字典存储，键为word，值为列表。
'''''''''
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
'''''
word_doc_freq={'word':n},n为多少篇文档出现过这个word。
'''''
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
'''''''''
word_id_map={'word':n}，n为索引，1、2、3、4.......vocab_size。
'''''''''
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)
'''''''''''
最后将vocab写入txt里。
'''''''''''
f = open('vocab_data/data_vocab.txt', 'w')
f.write(vocab_str)
f.close()

with open('vocab_data/train_doc_content_list.pk', 'wb+') as f:
    joblib.dump(train_doc_content_list, f)

with open('vocab_data/vocab.pk', 'wb+') as f:
    joblib.dump(vocab, f)

with open('vocab_data/train_shuffle_doc_label_list.pk', 'wb+') as f:
    joblib.dump(train_shuffle_doc_label_list, f)

with open('vocab_data/train_shuffle_doc_words_list.pk', 'wb+') as f:
    joblib.dump(train_shuffle_doc_words_list, f)

with open('vocab_data/test_shuffle_doc_label_list.pk', 'wb+') as f:
    joblib.dump(test_shuffle_doc_label_list, f)

with open('vocab_data/test_shuffle_doc_words_list.pk', 'wb+') as f:
    joblib.dump(test_shuffle_doc_words_list, f)

with open('vocab_data/test_doc_content_list.pk', 'wb+') as f:
    joblib.dump(test_doc_content_list, f)

with open('vocab_data/word_id_map.pk', 'wb+') as f:
    joblib.dump(word_id_map, f)

with open('vocab_data/word_doc_freq.pk', 'wb+') as f:
    joblib.dump(word_doc_freq, f)
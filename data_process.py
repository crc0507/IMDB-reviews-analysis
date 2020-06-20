from utils import *
from nltk.corpus import stopwords
import nltk
import glob
from tqdm import tqdm

txt_negfile = glob.glob('aclImdb/rain/neg/*.txt')
txt_posfile = glob.glob('aclImdb/train/pos/*.txt')

test_negfile = glob.glob('aclImdb/test/neg/*.txt')
test_posfile = glob.glob('aclImdb/test/pos/*.txt')

all_content = []
test_content = []

for filename in txt_negfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "neg"]
            all_content.append(content)

for filename in txt_posfile:
    with open(filename, 'r') as txt_posfile:
        buf2 = txt_posfile.readlines()
        for s in buf2:
            content = [s, "pos"]
            all_content.append(content)

for filename in test_negfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "neg"]
            test_content.append(content)

for filename in test_posfile:
    with open(filename, 'r') as txt_filtxt:
        buf1 = txt_filtxt.readlines()
        for s in buf1:
            content = [s, "pos"]
            test_content.append(content)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

with open("data/train_label.txt", "w") as f:
    for i in range(0, len(all_content)):
        f.write(all_content[i][1] + '\n')

with open("data/test_label.txt", "w") as f:
    for i in range(0, len(test_content)):
        f.write(test_content[i][1] + '\n')

train_clean_docs = []

for doc_content in tqdm(all_content):

    temp = clean_str(doc_content[0])
    words = temp.split()
    doc_words = []
    for word in words:
        if word not in stop_words:
            doc_words.append(word)
    train_doc_str = ' '.join(doc_words).strip()
    train_clean_docs.append(train_doc_str)

train_clean_corpus_str = '\n'.join(train_clean_docs)
with open("data/train_data_clean.txt", "w") as f:
    f.write(train_clean_corpus_str)

clean_docs = []
for doc_content in tqdm(test_content):
    temp = clean_str(doc_content[0])
    words = temp.split()
    doc_words = []
    for word in words:
        if word not in stop_words:
            doc_words.append(word)
    doc_str = ' '.join(doc_words).strip()
    clean_docs.append(doc_str)

clean_corpus_str = '\n'.join(clean_docs)
with open("data/test_data_clean.txt", "w") as f:
    f.write(clean_corpus_str)

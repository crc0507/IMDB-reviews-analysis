import os
import pandas as pd

def readfilelist(dir):
    filelist = []
    list = os.listdir(dir)
    print(list)
    print(len(list))
    for i in range(0,len(list)):
        path = os.path.join(dir,list[i])
        if os.path.isdir(path):
            filelist.extend(readfilelist(path))
        if os.path.isfile(path):
            filelist.append(path)
    return filelist

print('ok')
list = readfilelist('/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/train')
list1 = readfilelist('/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/test')
print("the lengh of list",len(list))

def readfile(list):
    new_sentences = []
    for e in list:
        f = open(e,'r')
        line = f.readline()
        new_sentences.append(line)
    return new_sentences

label = []
for i in range(25000):
    if i <= 12499:
        label.append(0)
    else:
        label.append(1)
print("the lengh of label:", len(label))

sentences = readfile(list)
sentences1 = readfile(list1)
print("the lengh of sentence:",len(sentences))
dataframe = pd.DataFrame({'text':sentences,'label':label})
dataframe1 = pd.DataFrame({'text':sentences1,'label':label})


dataframe.to_csv("/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/new_train.csv",index=False)
dataframe1.to_csv("/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/new_test.csv",index=False)
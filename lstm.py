import tensorflow as tf
from tensorflow.contrib.learn.python import learn
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences


MAX_DOCUMENT_LENGTH = 200
EMBEDDING_SIZE = 50

n_words=0


def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            x += line
    return x

def load_files(rootdir,label):
    list = os.listdir(rootdir)
    x=[]
    y=[]

    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            # print("Load file %s" % path)
            y.append(label)
            x.append(load_one_file(path))
    return x,y


def load_data():
    x=[]
    y=[]
    x1,y1=load_files("/Users/jizhimeicrc/Desktop/data/train/neg",0)
    x2,y2=load_files("/Users/jizhimeicrc/Desktop/data/train/pos", 1)
    x3,y3=load_files("/Users/jizhimeicrc/Desktop/data/test/neg", 0)
    x4, y4 = load_files("/Users/jizhimeicrc/Desktop/data/test/pos", 1)
    x=x1+x2+x3+x4
    y=y1+y2+y3+y4
    return x,y



def do_rnn(trainX, testX, trainY, testY):
    global n_words
    # Data preprocessing
    # Sequence padding
    print("GET n_words embedding %d" % n_words)


    trainX = pad_sequences(trainX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    testX = pad_sequences(testX, maxlen=MAX_DOCUMENT_LENGTH, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    print(trainX[:10])
    print(testX[:10])

    # Network building
    lstm = tflearn.input_data([None, MAX_DOCUMENT_LENGTH])
    lstm = tflearn.embedding(lstm, input_dim=n_words, output_dim=100)
    lstm = tflearn.lstm(lstm, 100, dropout=0.8)
    lstm = tflearn.fully_connected(lstm, 2, activation='softmax')
    lstm = tflearn.regression(lstm, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training



    model = tflearn.DNN(lstm, tensorboard_verbose=3)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
             batch_size=32, run_id="lstm")



def main(unused_argv):
    global n_words

    x,y=load_data()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

    vp = learn.preprocessing.VocabularyProcessor(max_document_length=MAX_DOCUMENT_LENGTH, min_frequency=1)
    vp.fit(x)
    x_train = np.array(list(vp.transform(x_train)))
    x_test = np.array(list(vp.transform(x_test)))
    n_words=len(vp.vocabulary_)
    print('Total words: %d' % n_words)

    do_rnn(x_train, x_test, y_train, y_test)



if __name__ == '__main__':
  tf.app.run()
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/train2.csv')
test_label_set = pd.read_csv('/Users/ssssshi/Desktop/Arlington/ML/project/aclImdb/test2.csv')
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_features = tfidf.fit_transform(label_set['text'])
test_features = tfidf.transform(test_label_set["text"])

#train_features = train_features.todense()
print(train_features)
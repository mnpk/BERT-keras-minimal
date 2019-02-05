'''
Train google BERT on Quora Insincere Questions Classification
https://www.kaggle.com/c/quora-insincere-questions-classification.
The model's task is to predict whether a question is sincere (label=0) or
insincere (label=1)
'''
import os
import numpy as np
import pandas as pd

from keras import Model
from keras.layers import Lambda, Dense
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn import metrics

from batch_generator.batch_generator import BatchGenerator
from load_pretrained_bert import load_google_bert

BERT_PRETRAINED_DIR = '../multi_cased_L-12_H-768_A-12/'
SEQ_LEN = 100
BATCH_SIZE = 16
LR = 1e-5

df = pd.read_csv("train.csv")
X = df["question_text"].values
Y = df['target'].values
print(X[0])  # How did Quebec nationalists see their province as a nation in the 1960s?
print(Y[0])  # 0
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, random_state=0)

train_gen = BatchGenerator(X_train,
                           vocab_file=os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt'),
                           seq_len=SEQ_LEN,
                           labels=Y_train,
                           do_lower_case=False,
                           batch_size=BATCH_SIZE)
valid_gen = BatchGenerator(X_valid,
                           vocab_file=os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt'),
                           seq_len=SEQ_LEN,
                           labels=Y_valid,
                           do_lower_case=False,
                           batch_size=BATCH_SIZE)

g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR, use_attn_mask=False, max_len=SEQ_LEN)
g_bert.summary()
# Choose Layer 0 as containing the features relevant for classification; see BERT paper for further explanation on
# this choice.
classification_features = Lambda(lambda x: x[:, 0, :])(g_bert.output)
out = Dense(1, activation='sigmoid')(classification_features)

model = Model(g_bert.inputs, out)
model.compile(optimizer=Adam(LR), loss=binary_crossentropy, metrics=['accuracy'])
model.summary()
model.fit_generator(train_gen,
                    epochs=1,
                    verbose=1,
                    validation_data=valid_gen,
                    shuffle=True)

Y_valid_predictions = model.predict_generator(valid_gen, verbose=1)
Y_valid = Y_valid[:len(Y_valid_predictions)]
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(Y_valid, (Y_valid_predictions > thresh).astype(int))
    print(f"F1 score at threshold {thresh} is {f1}")

'''
Best f1-score is .... at threshold of ...
Note that the results may vary slightly from run to run due to the non-deterministic nature of tensorflow/keras.
'''

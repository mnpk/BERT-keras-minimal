from keras.utils import Sequence
from keras_preprocessing.sequence import pad_sequences

import numpy as np

# FullTokenizer is the same as used by google bert
from batch_generator.tokenization import FullTokenizer


class BatchGenerator(Sequence):

    def __init__(self, texts, vocab_file, seq_len, do_lower_case=True, batch_size=16):
        # TODO: Are the BOS EOS etc. tokens?
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokizer = FullTokenizer(vocab_file, do_lower_case)
        self.padding_idx = self.tokizer.vocab['[PAD]']

        self.tokenized_texts = [self.tokizer.tokenize(text) for text in tqdm(texts)]
        print(self.tokenized_texts[0])
        self.tokenized_texts = [['[CLS]'] + text for text in tqdm(self.tokenized_texts)]

        self.text_as_sequence = [self.tokizer.convert_tokens_to_ids(tokens) for tokens in tqdm(self.tokenized_texts)]
        self.text_as_sequence = pad_sequences(self.text_as_sequence, maxlen=seq_len, padding='post',
                                              value=self.padding_idx)

    def __len__(self):
        return len(self.text_as_sequence)

    def __getitem__(self, idx):
        text_encoding = self.text_as_sequence[idx * self.batch_size: (idx + 1) * self.batch_size]
        # set segment as 0. The segment encoding is used, to mark the position of the sentence-next sentence
        # classification task. For fine-tuning, unless a similar task needs to be trained, we can use segemnt 0 only.
        segment_encoding = np.zeros((self.batch_size, self.seq_len))
        position_encoding = np.tile(list(range(self.seq_len)), (self.batch_size, 1))

        x = [text_encoding,
             segment_encoding,
             position_encoding]
        return x

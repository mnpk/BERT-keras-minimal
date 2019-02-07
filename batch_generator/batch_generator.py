from tqdm import tqdm
import numpy as np

from keras.utils import Sequence
from keras_preprocessing.sequence import pad_sequences

# FullTokenizer is the same as used by google bert
from batch_generator.tokenization import FullTokenizer


class BatchGenerator(Sequence):

    def __init__(self, texts, vocab_file, seq_len, labels=None, do_lower_case=True, batch_size=16):
        self.seq_len = seq_len
        self.labels = labels
        self.batch_size = batch_size

        self.tokizer = FullTokenizer(vocab_file, do_lower_case)
        self.padding_idx = self.tokizer.vocab['[PAD]']

        self.text_as_sequence = self._get_text_as_sequence(texts)

    def _get_text_as_sequence(self, texts):
        tokenized_texts = [['[CLS]'] + self.tokizer.tokenize(text) for text in tqdm(texts)]
        tokenized_texts = [text for text in tqdm(tokenized_texts)]
        text_as_sequence = [self.tokizer.convert_tokens_to_ids(tokens) for tokens in tqdm(tokenized_texts)]
        text_as_sequence = pad_sequences(text_as_sequence,
                                         maxlen=self.seq_len,
                                         padding='post', truncating='post',
                                         value=self.padding_idx)
        return text_as_sequence

    def __len__(self):
        return len(self.text_as_sequence) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size

        text_encoding = self.text_as_sequence[start: end]
        # set segment as 0. The segment encoding is used, to mark the position of the sentence-next sentence
        # classification task. For fine-tuning, unless a similar task needs to be trained, we can use segemnt 0 only.
        segment_encoding = np.zeros((self.batch_size, self.seq_len))
        position_encoding = np.tile(list(range(self.seq_len)), (self.batch_size, 1))

        x = [text_encoding,
             segment_encoding,
             position_encoding]
        if self.labels is None:
            return x
        else:
            return x, self.labels[start: end]

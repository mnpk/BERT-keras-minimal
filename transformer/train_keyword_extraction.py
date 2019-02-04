from keras import Model
from keras.layers import Dense, TimeDistributed

from transformer.model import create_transformer


def build_model():
    sequence_encoder_config = {
        'embedding_dim': 100,
        'vocab_size': 10_000,
        'max_len': 300,
        'trainable_pos_embedding': False,
        'num_heads': 2,
        'num_layers': 3,
        'd_hid': 12,
        'use_attn_mask': True
    }
    sequence_encoder = create_transformer(**sequence_encoder_config)
    dense = TimeDistributed(Dense(40_000, activation='softmax'), trainable=False)(sequence_encoder.output)
    model = Model(sequence_encoder.inputs, dense)
    return model


model = build_model()
model.summary()

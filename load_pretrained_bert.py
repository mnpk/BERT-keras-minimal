import keras
import numpy as np
import tensorflow as tf
import keras.backend as K

from transformer.bert_config import BertConfig
from transformer.layers import LayerNormalization, MultiHeadAttention, Gelu
from transformer.model import create_transformer


def load_google_bert(base_location: str = './google_bert/downloads/multilingual_L-12_H-768_A-12/',
                     use_attn_mask: bool = True, max_len: int = 512) -> keras.Model:
    bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
    assert bert_config.max_position_embeddings >= max_len, 'Maximal sequence length exceeds the length of the pretrained position embeddings.'
    init_checkpoint = base_location + 'bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,
                               vocab_size=bert_config.vocab_size, accurate_gelu=True, layer_norm_epsilon=1e-12,
                               max_len=max_len,
                               max_position_embeddings=bert_config.max_position_embeddings,
                               use_one_embedding_dropout=True, d_hid=bert_config.intermediate_size,
                               embedding_dim=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers,
                               num_heads=bert_config.num_attention_heads,
                               residual_dropout=bert_config.hidden_dropout_prob,
                               attention_dropout=bert_config.attention_probs_dropout_prob)
    weights = get_bert_weights_for_keras_model(check_point, model, var_names)
    model.set_weights(weights)
    return model


def get_bert_weights_for_keras_model(check_point, model, tf_var_names):
    keras_weights = [np.zeros(w.shape) for w in model.weights]
    keras_weights_set = []

    for var_name, _ in tf_var_names:
        qkv, unsqueeze, w_id = _get_tf2keras_weights_name_mapping(var_name)
        if w_id is None:
            # print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship
            pass
        else:
            # print(var_name, ' -> ', model.weights[w_id].name)
            keras_weights_set.append(w_id)
            keras_weight = keras_weights[w_id]
            tensorflow_weight = check_point.get_tensor(var_name)
            keras_weights[w_id] = _set_keras_weight_from_tf_weight(tensorflow_weight, keras_weight, qkv, unsqueeze, w_id)

    keras_layer_not_set = set(list(range(len(keras_weights)))) - set(keras_weights_set)
    assert len(keras_layer_not_set) == 0, 'Some weights were not set!'

    return keras_weights


def _set_keras_weight_from_tf_weight(tensorflow_weight, keras_weight, qkv, unsqueeze, w_id):
    if qkv is None:
        if w_id == 1 or w_id == 2:  # w_id==1: pos embedding, w_id==1:  word embedding
            keras_weight = tensorflow_weight
        else:
            keras_weight[:] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]
    else:
        p = {'q': 0, 'k': 1, 'v': 2}[qkv]
        if keras_weight.ndim == 3:
            dim_size = keras_weight.shape[1]
            keras_weight[0, :, p * dim_size:(p + 1) * dim_size] = tensorflow_weight if not unsqueeze else tensorflow_weight[None, ...]
        else:
            dim_size = keras_weight.shape[0] // 3
            keras_weight[p * dim_size:(p + 1) * dim_size] = tensorflow_weight

    return keras_weight


def _get_tf2keras_weights_name_mapping(var_name):
    w_id = None
    qkv = None
    unsqueeze = False

    var_name_splitted = var_name.split('/')
    if var_name_splitted[1] == 'embeddings':
        w_id = _get_embeddings_name(var_name_splitted)

    elif var_name_splitted[2].startswith('layer_'):
        qkv, unsqueeze, w_id = _get_layers_name(var_name_splitted)

    return qkv, unsqueeze, w_id


def _get_layers_name(var_name_splitted):
    first_vars_size = 5
    w_id = None
    qkv = None
    unsqueeze = False

    layer_number = int(var_name_splitted[2][len('layer_'):])
    if var_name_splitted[3] == 'attention':
        if var_name_splitted[-1] == 'beta':
            w_id = first_vars_size + layer_number * 12 + 5
        elif var_name_splitted[-1] == 'gamma':
            w_id = first_vars_size + layer_number * 12 + 4
        elif var_name_splitted[-2] == 'dense':
            if var_name_splitted[-1] == 'bias':
                w_id = first_vars_size + layer_number * 12 + 3
            elif var_name_splitted[-1] == 'kernel':
                w_id = first_vars_size + layer_number * 12 + 2
                unsqueeze = True
            else:
                raise ValueError()
        elif var_name_splitted[-2] == 'key' or var_name_splitted[-2] == 'query' or var_name_splitted[-2] == 'value':
            w_id = first_vars_size + layer_number * 12 + (0 if var_name_splitted[-1] == 'kernel' else 1)
            unsqueeze = var_name_splitted[-1] == 'kernel'
            qkv = var_name_splitted[-2][0]
        else:
            raise ValueError()
    elif var_name_splitted[3] == 'intermediate':
        if var_name_splitted[-1] == 'bias':
            w_id = first_vars_size + layer_number * 12 + 7
        elif var_name_splitted[-1] == 'kernel':
            w_id = first_vars_size + layer_number * 12 + 6
            unsqueeze = True
        else:
            raise ValueError()
    elif var_name_splitted[3] == 'output':
        if var_name_splitted[-1] == 'beta':
            w_id = first_vars_size + layer_number * 12 + 11
        elif var_name_splitted[-1] == 'gamma':
            w_id = first_vars_size + layer_number * 12 + 10
        elif var_name_splitted[-1] == 'bias':
            w_id = first_vars_size + layer_number * 12 + 9
        elif var_name_splitted[-1] == 'kernel':
            w_id = first_vars_size + layer_number * 12 + 8
            unsqueeze = True
        else:
            raise ValueError()
    return qkv, unsqueeze, w_id


def _get_embeddings_name(parts):
    n = parts[-1]
    if n == 'token_type_embeddings':
        w_id = 0
    elif n == 'position_embeddings':
        w_id = 1
    elif n == 'word_embeddings':
        w_id = 2
    elif n == 'gamma':
        w_id = 3
    elif n == 'beta':
        w_id = 4
    else:
        raise ValueError()
    return w_id


if __name__ == '__main__':
    BERT_PRETRAINED_DIR = '../../multi_cased_L-12_H-768_A-12/'
    g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR, use_attn_mask=False, max_len=128)
    g_bert.summary()
    g_bert.save('bert_multi_cased_l_12_h_768_a_12.hdf5')
    ############################################################################################
    K.clear_session()
    model = keras.models.load_model('bert_multi_cased_l_12_h_768_a_12.hdf5',
                                    custom_objects={'LayerNormalization': LayerNormalization,
                                                    'MultiHeadAttention': MultiHeadAttention,
                                                    'Gelu': Gelu})
    model.summary()
    ############################################################################################
    # We can load a trained model which was trained on sequences of length 128 into a model which has
    # longer sequence length. This is possible, since the position embeddings in load_google_bert
    # are from 0-512, as defined in the BERT config. For optimal results, the positional embeddings should be
    # frozen during training, such that the model can extrapolate sequences with length larger than 128
    K.clear_session()
    g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR, use_attn_mask=False, max_len=512)
    g_bert.load_weights('bert_multi_cased_l_12_h_768_a_12.hdf5')
    g_bert.summary()

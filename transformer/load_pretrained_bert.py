import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
from google_bert.modeling import BertConfig
from transformer.layers import LayerNormalization, MultiHeadAttention, Gelu
from transformer.model import create_transformer


def load_google_bert(base_location: str = './google_bert/downloads/multilingual_L-12_H-768_A-12/',
                     use_attn_mask: bool = True, max_len: int = 512) -> keras.Model:
    bert_config = BertConfig.from_json_file(base_location + 'bert_config.json')
    init_checkpoint = base_location + 'bert_model.ckpt'
    var_names = tf.train.list_variables(init_checkpoint)
    check_point = tf.train.load_checkpoint(init_checkpoint)
    model = create_transformer(embedding_layer_norm=True, neg_inf=-10000.0, use_attn_mask=use_attn_mask,
                               vocab_size=bert_config.vocab_size, accurate_gelu=True, layer_norm_epsilon=1e-12, max_len=max_len,
                               use_one_embedding_dropout=True, d_hid=bert_config.intermediate_size,
                               embedding_dim=bert_config.hidden_size, num_layers=bert_config.num_hidden_layers,
                               num_heads=bert_config.num_attention_heads,
                               residual_dropout=bert_config.hidden_dropout_prob,
                               attention_dropout=bert_config.attention_probs_dropout_prob)
    weights = get_bert_weights_for_keras_model(check_point, max_len, model, var_names)
    model.set_weights(weights)
    return model


def get_bert_weights_for_keras_model(check_point, max_len, model, tf_var_names):
    keras_weights = [np.zeros(w.shape) for w in model.weights]
    keras_weights_set = []

    for var_name, _ in tf_var_names:
        qkv, unsqueeze, w_id = _get_tf2keras_weights_name_mapping(var_name)
        if w_id is None:
            print('not mapped: ', var_name)  # TODO pooler, cls/predictions, cls/seq_relationship
        else:
            print(var_name, ' -> ', model.weights[w_id].name)
            keras_weights_set.append(w_id)
            keras_weight = keras_weights[w_id]
            tensorflow_weight = check_point.get_tensor(var_name)
            keras_weights[w_id] = _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id)

    print(set(list(range(len(keras_weights)))) - set(keras_weights_set))
    return keras_weights


def _set_keras_weight_from_tf_weight(max_len, tensorflow_weight, keras_weight, qkv, unsqueeze, w_id):
    if qkv is None:
        if w_id == 1:  # pos embedding
            keras_weight[:max_len, :] = tensorflow_weight[:max_len, :] if not unsqueeze else tensorflow_weight[None, :max_len, :]

        elif w_id == 2:  # word embedding
            keras_weight = tensorflow_weight
            # # ours: unk, [vocab], pad, msk(mask), bos(cls), del(use sep again), eos(sep)
            # # theirs: pad, 99 unused, unk, cls, sep, mask, [vocab]
            # saved = tensorflow_weight  # vocab_size, emb_size
            # # weights[our_position] = saved[their_position]
            # keras_weight[0] = saved[1 + TextEncoder.BERT_UNUSED_COUNT]  # unk
            # keras_weight[1:vocab_size] = saved[-vocab_size + 1:]
            # keras_weight[vocab_size + TextEncoder.PAD_OFFSET] = saved[0]
            # keras_weight[vocab_size + TextEncoder.MSK_OFFSET] = saved[4 + TextEncoder.BERT_UNUSED_COUNT]
            # keras_weight[vocab_size + TextEncoder.BOS_OFFSET] = saved[2 + TextEncoder.BERT_UNUSED_COUNT]
            # keras_weight[vocab_size + TextEncoder.DEL_OFFSET] = saved[3 + TextEncoder.BERT_UNUSED_COUNT]
            # keras_weight[vocab_size + TextEncoder.EOS_OFFSET] = saved[3 + TextEncoder.BERT_UNUSED_COUNT]
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

    BERT_PRETRAINED_DIR = '../../multi_cased_L-12_H-768_A-12'
    # vocab_size: 119547
    g_bert = load_google_bert(base_location=BERT_PRETRAINED_DIR + '/', use_attn_mask=False)
    g_bert.summary()
    #g_bert.save('bert_multi_cased_l_12_h_768_a_12.hdf5')
    #############################################################################################
    # K.clear_session()
    # model = keras.models.load_model('bert_multi_cased_l_12_h_768_a_12.hdf5',
    #                                 custom_objects={'LayerNormalization': LayerNormalization,
    #                                                 'MultiHeadAttention': MultiHeadAttention,
    #                                                 'Gelu': Gelu})
    # model.summary()
    '''
    
    x contains:
    x[0]: Tokenized text as input. TODO: left or right pad?
    x[0].shape = (batch_size, seq_len)
    x[1]: Segment input. keep this as 0
    x[1].shape = (batch_size, seq_len)
    x[2]: Position input. Positional encoding of the form: pos_0 = 0; pos_1 = 1, etc.
    x[2].shape = (batch_size, seq_len)
    if use_attn_mask:
        x[3]: the input attention mask, default set to np.ones. see create_attention_mask_from_input_mask in tf-bert modeling.py
        x[3].shape = (batch_size, 1, seq_len, seq_len)
    '''

    x = [np.random.randint(0, 100_000, (1, 512)), np.random.randint(0, 1, (1, 512)), np.random.randint(0, 511, (1, 512))]#, np.ones((1, 1, 512, 512))]
    for item in x:
        print(item.shape)
    print(g_bert.inputs)
    print(g_bert.output)
    preds = g_bert.predict(x)
    print(preds.shape)
    #############################################################################################
    K.clear_session()
    '''
    segment_input === token_type_embeddings, 2 values 2*768=1536
    position_input == position_embeddings, 512 values, 512*768=393216
    token_input == word_embeddings! 119547*768 = 91812096 (91736832=119449*768)
    '''

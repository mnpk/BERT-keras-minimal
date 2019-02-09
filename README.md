 **Refactored version of Separius/BERT-keras which focuses on providing the two functionalities:**
 
 * _Load a pretrained tensorflow BERT model (load_pretrained_bert.py) and use it as a usual keras model_
 * _Provide a batch_generator (batch_generator.py) which provides the required input for the keras model (text tokenization, positional encoding and segment encoding)._
 
* In contrast to the implementation of Separius, we keep the vocabulary ordering of the original tensoflow implementation.

See example_kaggle_quora_insincere.py for a simple example.

 **Load google BERT from a tensorflow model:**
load_google_bert loads the weights from a pretrained tensorflow checkpoint. In particular, it loads
all position embeddings weights, even if the keras model has a shorter sequence length. It is therefore possible to train
a model with e.g. a sequence length of 70, with positional embeddings frozen and then to load the model for inference
with a larger sequence length.
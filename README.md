 **Refactored version of Separius/BERT-keras which focuses on providing the two functionalities:**
 
 * _Load a pretrained tensorflow BERT model (load_pretrained_bert.py) and use it as a usual keras model_
 * _Provide a batch_generator (batch_generator.py) which provides the required input for the keras model (text tokenization, positional encoding and segment encoding)._
 
* In contrast to the implementation of Separius, we keep the vocabulary ordering of the original tensoflow implementation.

See example_kaggle_quora_insincere.py for a simple example.
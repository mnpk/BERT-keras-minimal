 **Refactored version of Separius/BERT-keras which focuses on providing the two functionalities:**
 
 * _Load a pretrained tensorflow BERT model (load_pretrained_bert.py) and save it as a keras model_
 * _Provide a batch_generator (batch_generator.py) which provides the required input for the keras model (text tokenization, positional encoding and segment encoding)._
 
* In contrast to the implementation of Separius, we keep the vocabulary ordering of the original tensoflow implementation.
* The code in this repository is minimal, no explicit TPU support, no functionality for loading OPENAI models.

This is work in progress.
# DistilBERT implementation

This provides some background on how the DistBERT model has been implemented from [sentence transformers](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2).

## Source analysis

The entrypoint for generic model instantiation is the [Sentence Transformer](https://github.com/UKPLab/sentence-transformers/blob/61806f0e1085f000dfbf4e586074b4250986cf39/sentence_transformers/SentenceTransformer.py#L33) object.

This class will read the [`modules.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/modules.json) configuration file in HuggingFace repo. This defines 3 layers:

* A [generic transformer](https://github.com/UKPLab/sentence-transformers/blob/2158fff3aa96651b10fe367c41fdd5008a33c5c6/sentence_transformers/models/Transformer.py#L8) with parameters defined in [`sentence_bert_config.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/sentence_bert_config.json).
  It will automatically load the transformer implementation from the [`config.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/config.json).
* A [pooling layer](https://github.com/UKPLab/sentence-transformers/blob/2158fff3aa96651b10fe367c41fdd5008a33c5c6/sentence_transformers/models/Pooling.py#L9) with config [`1_Pooling/config.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/1_Pooling/config.json).
* A [dense layer](https://github.com/UKPLab/sentence-transformers/blob/2158fff3aa96651b10fe367c41fdd5008a33c5c6/sentence_transformers/models/Dense.py#L11) with config [`2_Dense/config.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/2_Dense/config.json)

The transformer [`config.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/config.json) points to the class [`DistilBertModel`](https://github.com/huggingface/transformers/blob/f71fb5c36e739d8224419bb091b4c16531df829f/src/transformers/models/distilbert/modeling_distilbert.py#L435) of the HuggingFace transformers lib.

The "fast" tokenizer using the HuggingFace library [`tokenizers`](https://github.com/huggingface/tokenizers) is build for Bert in the file [`transformers.convert_slow_tokenizer`](https://github.com/huggingface/transformers/blob/27b3031de2fb8195dec9bc2093e3e70bdb1c4bff/src/transformers/convert_slow_tokenizer.py#L72) with parameters defined in [`tokenizer.json`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2/blob/main/tokenizer.json).

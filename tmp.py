from sentence_transformers import SentenceTransformer

from mlmodule.contrib.sentences.distilbert.config import DistilBertConfig
from mlmodule.contrib.sentences.distilbert.modules import DistilBertModule
from mlmodule.contrib.sentences.distilbert.tokenizers import get_distil_bert_tokenizer
from mlmodule.contrib.sentences.distilbert.transforms import TokenizerTransform

SENTENCE = "Hello world"

st = SentenceTransformer(
    "distiluse-base-multilingual-cased-v2",
    device="cpu",  # We set CPU here to not automatically load the model on GPU
)

print(st.encode([SENTENCE]))

tokenizer = TokenizerTransform(get_distil_bert_tokenizer())
print(DistilBertModule(DistilBertConfig()).forward(*tokenizer(SENTENCE)))

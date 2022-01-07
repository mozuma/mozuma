from typing import Callable, List, Optional

import torch.nn
from tokenizers import Tokenizer

from mlmodule.contrib.sentences.distilbert.tokenizers import get_distil_bert_tokenizer
from mlmodule.contrib.sentences.distilbert.transforms import TokenizerTransform
from mlmodule.v2.torch.modules import TorchMlModule

from .blocks import Embeddings, Transformer
from .config import DistilBertConfig


class DistilBertModule(TorchMlModule):
    _tokenizer: Optional[Tokenizer] = None

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.config = config

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

    @classmethod
    def get_tokenizer(cls) -> Tokenizer:
        """Return a tokenizer loaded once and cached on the instance"""
        if cls._tokenizer is None:
            cls._tokenizer = get_distil_bert_tokenizer()
        return cls._tokenizer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`DistilBertTokenizer`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for
                details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `(batch_size)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated
                vectors than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
                See `attentions` under returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers.
                See `hidden_states` under returned tensors for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_dataset_transforms(self) -> List[Callable]:
        return [TokenizerTransform(self.get_tokenizer())]

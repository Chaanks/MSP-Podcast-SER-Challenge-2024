import torch
import logging
import torch.nn.functional as F
import os

from speechbrain.lobes.models.huggingface_transformers.huggingface import (
    HFTransformersInterface,
)

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RoBERTa(HFTransformersInterface):

    def __init__(
        self,
        source,
        save_path,
        freeze=True,
        output_norm=True,
        output_all_hiddens=False,
    ):
        super().__init__(
            source=source,
            save_path=save_path,
            freeze=freeze,
        )

        self.load_tokenizer(source=source)
        self.output_norm = output_norm
        self.output_all_hiddens = output_all_hiddens

    def forward(self, text, text_len=None):
        # If we freeze, we simply remove all grads from the graph.
        if self.freeze:
            with torch.no_grad():
                return self.extract_features(text, text_len)

        return self.extract_features(text, text_len)

    def extract_features(self, text, text_len=None):
        # Tokenize the input text before feeding
        input_texts = self.tokenizer(text, return_tensors="pt", padding=True)
        input_lengths = torch.sum(input_texts.input_ids != 1, dim=1)

        # Set the right device for the input.
        for key in input_texts.keys():
            input_texts[key] = input_texts[key].to(device=self.model.device)
            input_texts[key].requires_grad = False

        out = self.model(
            **input_texts, output_hidden_states=self.output_all_hiddens
        )

        if self.output_all_hiddens:
            out = torch.stack(list(out.hidden_states), dim=0)
            norm_shape = out.shape[-3:]
        else:
            out = out.last_hidden_state
            norm_shape = out.shape

        # We normalize the output if required
        if self.output_norm:
            out = F.layer_norm(out, norm_shape[1:])

        return out, input_lengths

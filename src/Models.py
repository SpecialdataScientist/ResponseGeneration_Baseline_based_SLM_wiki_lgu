import torch
import torch.nn as nn

from transformers import BartConfig, BartForConditionalGeneration


class GeneratorModel(nn.Module):
    def __init__(self, args):
        super(GeneratorModel, self).__init__()
        self.args = args
        self.t5_config = BartConfig()

        self.generator_model = BartForConditionalGeneration.from_pretrained(args.transformers_generator_model_name)

    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask,
                labels=None, target=None, num_beams=1, num_return_sequences=1, no_repeat_ngram_size=3,
                gen_max_label_len=512, return_dict=None):

        if self.training:
            output = self.generator_model(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                          decoder_input_ids=decoder_input_ids,
                                          decoder_attention_mask=decoder_attention_mask,
                                          labels=target)

            return {'logits': output.logits, 'loss': output.loss.mean()}

        output = self.generator_model.generate(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask,
                                               num_beams=num_beams, num_return_sequences=num_return_sequences,
                                               no_repeat_ngram_size=no_repeat_ngram_size, max_length=gen_max_label_len)

        return {'pred': output, 'label': labels}
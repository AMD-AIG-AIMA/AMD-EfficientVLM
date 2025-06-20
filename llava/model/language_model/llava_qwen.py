from typing import List, Optional, Tuple, Union, Dict, Callable
import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoModelForCausalLM 

from ..modeling_qwen import Qwen2VLoRAModel, Qwen2VLoRAForCausalLM

from llava.constants import IMAGE_TOKEN_INDEX
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..llava_arch import LlavaVLoRAMetaModel, LlavaVLoRAMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwen2VLoRAConfig(Qwen2Config):
    model_type = "llava_qwen_vlora"


class LlavaQwen2VLoRAModel(LlavaVLoRAMetaModel, Qwen2VLoRAModel):
    config_class = LlavaQwen2VLoRAConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwen2VLoRAModel, self).__init__(config)


class LlavaQwen2VLoRAForCausalLM(Qwen2VLoRAForCausalLM, LlavaVLoRAMetaForCausalLM):
    config_class = LlavaQwen2VLoRAConfig

    def __init__(self, config):
        # super(Qwen2VLoRAForCausalLM, self).__init__(config)
        Qwen2VLoRAForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen_vlora"
        config.rope_scaling = None

        self.model = LlavaQwen2VLoRAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def get_vlora_weights(self, images, input_ids):
        if images is None:
            return None
        vlora_weights = self.encode_images(images)

        # fill zero for language-only data
        mask = input_ids == IMAGE_TOKEN_INDEX
        mask = (mask.sum(1) > 0).long().reshape(-1, 1, 1)
        for vlora_weights_sub in vlora_weights:
            for key in vlora_weights_sub:
                if vlora_weights_sub[key] != (None, None):
                    vlora_weights_sub[key] = (vlora_weights_sub[key][0] * mask, 
                                              vlora_weights_sub[key][1])

        return vlora_weights

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        vlora_weights = None,
        img_token_idx = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if vlora_weights is None:
            vlora_weights = self.get_vlora_weights(images, input_ids)
        
        assert vlora_weights is not None

        # TODO: an ugly way to handle IMAGE_TOKEN_INDEX
        mask = input_ids == IMAGE_TOKEN_INDEX
        input_ids[mask] = 0
        if mask.shape != attention_mask.shape:  # not equal when .generate()
            assert img_token_idx is not None
            for i in range(img_token_idx.shape[0]):
                if img_token_idx[i] >= 0:
                    attention_mask[i, img_token_idx[i]] = 0
        else:
            attention_mask[mask] = 0

        # if input_ids is None:
        #     print("input_ids is None")
        # else:
        #     print("Hello world")

        # if inputs_embeds is None:
        #     print("inputs_embeds is None.\n")
            # (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                vlora_weights=vlora_weights,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        vlora_weights = self.get_vlora_weights(images, inputs)

        img_token_idx_list = []
        for i in range(inputs.shape[0]):
            mask = (inputs[i] == IMAGE_TOKEN_INDEX).int()
            if mask.sum() > 0:
                img_token_idx_list.append(mask.argmax().item())
            else:
                img_token_idx_list.append(-1)
        img_token_idx = torch.Tensor(img_token_idx_list).long().to(images.device)

        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, 
                                vlora_weights=vlora_weights, img_token_idx=img_token_idx, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        vlora_weights = kwargs.pop("vlora_weights", None)
        img_token_idx = kwargs.pop("img_token_idx", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if vlora_weights is not None:
            inputs['vlora_weights'] = vlora_weights
        if img_token_idx is not None:
            inputs['img_token_idx'] = img_token_idx
        return inputs


AutoConfig.register("llava_qwen_vlora", LlavaQwen2VLoRAConfig)
AutoModelForCausalLM.register(LlavaQwen2VLoRAConfig, LlavaQwen2VLoRAForCausalLM)

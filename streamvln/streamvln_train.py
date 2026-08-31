# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ast
import logging
import pathlib
from typing import Dict
from PIL import ImageFile
from functools import partial

import re
import torch
from torchvision.transforms import v2

import transformers

from transformers import AutoConfig
from llava.train.llava_trainer import LLaVATrainer

# Custom Trainer with world model loss logging
class StreamVLNTrainer(LLaVATrainer):
    """Custom Trainer for StreamVLN with world model loss logging
    
    Follows HuggingFace Trainer (trainer.py):
    - Uses accumulation similar to _total_loss_scalar
    - Accumulates scalar metrics in training_step
    - Averages in log (see _maybe_log_save_evaluate)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Accumulation vars like Trainer._total_loss_scalar
        self._total_nav_loss_scalar = 0.0
        self._total_wm_loss_scalar = 0.0
        self._total_wm_2d_loss_scalar = 0.0
        self._total_wm_3d_loss_scalar = 0.0
        self._custom_metrics_count = 0  # Actual accumulation count (gradient_accumulation_steps)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Standard HuggingFace pattern: compute and return loss only.
        See trainer.py L3838-3921.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """
        See trainer.py L3767-3838.
        Extends training_step to accumulate extra metrics.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, return_outputs=True)
            
        # loss is tuple: (loss_tensor, outputs)
        if isinstance(loss, tuple):
            loss_tensor, outputs = loss
            loss = loss_tensor
        else:
            outputs = None

        kwargs = {}
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        # Accumulate extra metrics (like _total_loss_scalar in trainer.py)
        if outputs is not None:
            if hasattr(outputs, 'nav_loss'):
                self._total_nav_loss_scalar += outputs.nav_loss.detach().item()
            if hasattr(outputs, 'wm_loss'):
                self._total_wm_loss_scalar += outputs.wm_loss.detach().item()
            if hasattr(outputs, 'wm_2d_loss'):
                self._total_wm_2d_loss_scalar += outputs.wm_2d_loss.detach().item()
            if hasattr(outputs, 'wm_3d_loss'):
                self._total_wm_3d_loss_scalar += outputs.wm_3d_loss.detach().item()
            # Increment count after each successful accumulation
            self._custom_metrics_count += 1

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def log(self, logs):
        """
        See trainer.py L3555-3581 and _maybe_log_save_evaluate L2987-3028.
        Adds averaged extra metrics when Trainer calls log.
        """
        # Average using actual accumulation count (gradient_accumulation_steps)
        if self._custom_metrics_count > 0:
            # Add averaged world model losses
            if self._total_nav_loss_scalar > 0:
                logs["nav_loss"] = round(self._total_nav_loss_scalar / self._custom_metrics_count, 4)
            if self._total_wm_loss_scalar > 0:
                logs["wm_loss"] = round(self._total_wm_loss_scalar / self._custom_metrics_count, 4)
            if self._total_wm_2d_loss_scalar > 0:
                logs["wm_2d_loss"] = round(self._total_wm_2d_loss_scalar / self._custom_metrics_count, 4)
            if self._total_wm_3d_loss_scalar > 0:
                logs["wm_3d_loss"] = round(self._total_wm_3d_loss_scalar / self._custom_metrics_count, 4)
        
        # Call parent log (triggers callbacks and wandb/tensorboard)
        super().log(logs)
        
        # Reset accumulators (like tr_loss -= tr_loss in _maybe_log_save_evaluate)
        self._total_nav_loss_scalar = 0.0
        self._total_wm_loss_scalar = 0.0
        self._total_wm_2d_loss_scalar = 0.0
        self._total_wm_3d_loss_scalar = 0.0
        self._custom_metrics_count = 0

from llava import conversation as conversation_lib
from llava.model import *
from llava.utils import rank0_print

from streamvln.model.stream_video_vln import StreamVLNForCausalLM
from streamvln.dataset.vln_action_dataset import collate_fn, VLNActionDataset

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

from streamvln.args import ModelArguments, DataArguments, TrainingArguments

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

        
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["vision_tower",
                            "mm_projector",
                            "mem_projector",
                            "point_projector",
                            "vision_resampler",
                            "mem_resampler",
                            "pointnet"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    torch.cuda.synchronize()
    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    # if 'mm_vision_resampler' in trainer.args.mm_tunable_parts:
    #     keys_to_match = ["mm_projector", "vision_resampler"]
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(["embed_tokens", "embed_in"])

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split("/")[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith("checkpoint-"):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
                
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def safe_save_model_for_hf_trainer_fsdp(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.state_dict_type = "FULL_STATE_DICT"
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, vision_tower, data_args) -> Dict:
    train_dataset = VLNActionDataset(tokenizer=tokenizer, data_args=data_args, task_id=0)
    rank0_print('len train_dataset ', len(train_dataset))
    data_collator = partial(collate_fn, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args):
    # import ipdb; ipdb.set_trace()
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

    # import ipdb; ipdb.set_trace()
    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode
    if model_args.mm_spatial_pool_size is not None:
        overwrite_config["mm_spatial_pool_size"] = model_args.mm_spatial_pool_size

    if data_args.num_future_steps:
        overwrite_config["num_future_steps"] = data_args.num_future_steps
    if data_args.num_history:
        overwrite_config["num_history"] = data_args.num_history
    
    # World Query Token config (v2 - Transformer Decoder)
    overwrite_config["use_world_query"] = model_args.use_world_query
    if hasattr(model_args, 'num_world_query_tokens'):
        overwrite_config["num_world_query_tokens"] = model_args.num_world_query_tokens
    if hasattr(model_args, 'world_query_decoder_depth'):
        overwrite_config["world_query_decoder_depth"] = model_args.world_query_decoder_depth
    if hasattr(model_args, 'wm_loss_weight') and model_args.wm_loss_weight is not None:
        overwrite_config["wm_loss_weight"] = model_args.wm_loss_weight
    if hasattr(model_args, 'wm_2d_loss_ratio') and model_args.wm_2d_loss_ratio is not None:
        overwrite_config["wm_2d_loss_ratio"] = model_args.wm_2d_loss_ratio
    if hasattr(model_args, 'wm_use_2d_loss'):
        overwrite_config["wm_use_2d_loss"] = model_args.wm_use_2d_loss
    if hasattr(model_args, 'wm_use_3d_loss'):
        overwrite_config["wm_use_3d_loss"] = model_args.wm_use_3d_loss
    if hasattr(model_args, 'wm_2d_loss_type'):
        overwrite_config["wm_2d_loss_type"] = model_args.wm_2d_loss_type

    if model_args.mm_tunable_parts:
        overwrite_config["mm_tunable_parts"] = model_args.mm_tunable_parts
    
    overwrite_config["mm_newline_position"] = model_args.mm_newline_position
    overwrite_config["mm_patch_merge_type"] = model_args.mm_patch_merge_type
   
    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    model = StreamVLNForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
                )
    
    # Log attention implementation info
    rank0_print(f"\n{'='*60}")
    rank0_print(f"Attention implementation:")
    rank0_print(f"   - Implementation: {training_args.attn_implementation}")
    if training_args.attn_implementation == "sdpa":
        rank0_print(f"   - Benefits: custom 4D mask + performance tuning")
        rank0_print(f"   - PyTorch picks best kernel (Flash/xFormers/Native)")
    if hasattr(model.config, 'use_world_query') and model.config.use_world_query:
        rank0_print(f"   - World Query: enabled")
        rank0_print(f"   - 4D Attention Mask: active")
    rank0_print(f"{'='*60}\n")
    
    return model


def train(attn_implementation=None):
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
    # import ipdb; ipdb.set_trace()
    model = get_model(model_args, training_args, data_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # import ipdb; ipdb.set_trace()
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            # target_modules=find_all_linear_names(model, training_args.lora_target_modules.split(",")),
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        # import ipdb; ipdb.set_trace()

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        # Default: load tokenizer directly from checkpoint directory
        # AutoTokenizer loads tokenizer.json from checkpoint if present
        rank0_print(f"Loading tokenizer from checkpoint: {model_args.model_name_or_path}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], "patch_size should be in [224, 336, 384, 448, 512]"
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts and training_args.token_compression=="resampler":
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            
            # if "mm_language_model" in tunable_parts:
            #     for name, param in model.named_parameters():
            #         if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
            #             param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                language_model_prefixes = (
                    "model.embed_tokens",
                    "model.layers",
                    "model.norm",
                    "lm_head",
                )
                for name, param in model.named_parameters():
                    if name.startswith(language_model_prefixes):
                        param.requires_grad_(True)
                        
            if "mm_lora_layer" in tunable_parts:
                for name, param in model.named_parameters():
                    if "lora" in name:
                        param.requires_grad_(True)
        
            if "fusion_block" in tunable_parts:
                # Unfreeze cross-attention fusion modules (excludes world model)
                fusion_module_names = [
                    "clip_norm", "spatial_norm", 
                    "clip_query_proj", "spatial_key_proj", "spatial_value_proj",
                    "cross_attention", "out_norm", "out_proj", "fusion_dropout"
                ]
                for name, param in model.named_parameters():
                    if any(module_name in name for module_name in fusion_module_names):
                        param.requires_grad_(True)
            
            if "world_model" in tunable_parts:
                # Unfreeze world query modules (Transformer Decoder)
                world_model_names = [
                    "world_query_2d", "world_query_3d",
                    "world_query_2d_projector", "world_query_3d_projector",
                    "world_mask_token_2d", "world_mask_token_3d",
                    "world_decoder_2d", "world_decoder_3d",
                    "world_decoder_2d_norm", "world_decoder_3d_norm",
                    "world_decoder_2d_pred", "world_decoder_3d_pred"
                ]
                for name, param in model.named_parameters():
                    if any(module_name in name for module_name in world_model_names):
                        param.requires_grad_(True)

        rank0_print("Training parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:  # Check if the parameter requires training
                rank0_print(name)
        rank0_print("="*80)
        
        # Summarize trainable modules (group by first 2-3 name segments)
        from collections import defaultdict
        module_stats = defaultdict(lambda: {"trainable": 0, "frozen": 0})
        
        for name, param in model.named_parameters():
            # Take first 2-3 module name segments
            parts = name.split('.')
            if len(parts) >= 3:
                module_key = '.'.join(parts[:3])
            elif len(parts) >= 2:
                module_key = '.'.join(parts[:2])
            else:
                module_key = parts[0]
            
            param_count = param.ds_numel if hasattr(param, "ds_numel") else param.numel()
            if param.requires_grad:
                module_stats[module_key]["trainable"] += param_count
            else:
                module_stats[module_key]["frozen"] += param_count
        
        rank0_print("\n" + "="*80)
        rank0_print("Module trainability summary (first 2-3 levels):")
        rank0_print("="*80)
        
        trainable_modules = []
        frozen_modules = []
        
        for module_key in sorted(module_stats.keys()):
            stats = module_stats[module_key]
            total = stats["trainable"] + stats["frozen"]
            trainable_ratio = stats["trainable"] / total * 100 if total > 0 else 0
            
            if stats["trainable"] > 0:
                trainable_modules.append(
                    f"  + {module_key:<45} | trainable: {stats['trainable']/1e6:>8.2f}M ({trainable_ratio:>5.1f}%)"
                )
            else:
                frozen_modules.append(
                    f"  - {module_key:<45} | frozen:   {stats['frozen']/1e6:>8.2f}M"
                )
        
        if trainable_modules:
            rank0_print("\n[Trainable modules]:")
            for line in trainable_modules:
                rank0_print(line)
        
        if frozen_modules:
            rank0_print("\n[Frozen modules]:")
            for line in frozen_modules:
                rank0_print(line)
        
        rank0_print("="*80 + "\n")
        
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if data_args.data_augmentation:
        data_args.transform_train = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(brightness=0.2, saturation=0.2),
            v2.RandomPosterize(bits=4),
            v2.RandomAdjustSharpness(sharpness_factor=1.5),
            v2.RandomAutocontrast(),
            v2.ToPILImage()
        ])
    else:
        data_args.transform_train = None

    # import ipdb; ipdb.set_trace()
    data_module = make_supervised_data_module(tokenizer=tokenizer,vision_tower=vision_tower, data_args=data_args)
    
    params_no_grad = [
        n for n, p in model.named_parameters() if not p.requires_grad
    ]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'
                    .format(len(params_no_grad), params_no_grad))
            else:
                print(
                    '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'
                    .format(len(params_no_grad),
                            ', '.join(params_no_grad[:10])))
            print(
                "[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental."
            )
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining"
            )
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('ignored_parameters', True)
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args,
                                **kwargs,
                                use_orig_params=use_orig_params)
                return wrap_func
            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    
    trainer = StreamVLNTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    
    # ========== Compute steps_per_epoch and step-based training ==========
    train_dataset = data_module['train_dataset']
    dataset_size = len(train_dataset)
    
    # Effective batch size per GPU
    per_device_batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    num_gpus = 8 #<TODO> set to number of GPUs used for training
    
    # Compute steps_per_epoch
    # Note: with group_by_task=True, each batch may contain one task only
    # but steps still use total dataset size
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
    steps_per_epoch = dataset_size // effective_batch_size
    if dataset_size % effective_batch_size != 0:
        steps_per_epoch += 1  # ceil division
    
    rank0_print(f"Dataset statistics:")
    rank0_print(f"   - Dataset size: {dataset_size:,} samples")
    rank0_print(f"   - Per-GPU batch size: {per_device_batch_size}")
    rank0_print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    rank0_print(f"   - GPU count: {num_gpus}")
    rank0_print(f"   - Effective batch size: {effective_batch_size}")
    rank0_print(f"   - Steps per epoch: {steps_per_epoch:,}")
    
    # If num_train_epochs=1 and max_steps unset, auto-compute max_steps
    if training_args.num_train_epochs == 1.0 and training_args.max_steps == -1:
        max_steps = steps_per_epoch
        training_args.max_steps = max_steps
        rank0_print(f"   - Auto-set max_steps = {max_steps:,} (equivalent to 1 epoch)")
    
    # If save_strategy is "epoch", switch to "steps" for more frequent checkpoints
    if training_args.save_strategy == "epoch":
        # save_steps = max(500, steps_per_epoch // 10)
        save_steps = max(500, steps_per_epoch // 10)
        training_args.save_strategy = "steps"
        training_args.save_steps = save_steps
        rank0_print(f"   - Save strategy set to 'steps', every {save_steps:,} steps")
    
    rank0_print("")
    
    # ========== Resume training ==========
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        latest = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
        rank0_print(f"Found existing checkpoint: {latest.name}, resuming training...")
        trainer.train(resume_from_checkpoint=True)  # True: auto-use latest checkpoint in output_dir
    else:
        rank0_print("Starting new training...")
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        # import ipdb; ipdb.set_trace()
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
    else:
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if training_args.fsdp:
            safe_save_model_for_hf_trainer_fsdp(trainer=trainer, output_dir=training_args.output_dir)
        else:
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()

"""
StreamVLN Model with World Query Tokens for Future Frame Prediction.

Uses CUT3R spatial encoder + SigLIP cross-attention fusion and latent
world-model prediction of future 2D (SigLIP) and 3D (CUT3R) features.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from math import ceil
from typing import List, Optional, Union, Tuple
from functools import partial
from timm.models.vision_transformer import Block

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2ForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenModel
from llava.model.llava_arch import LlavaMetaForCausalLM
from ..utils.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, MEMORY_TOKEN_INDEX
from .cut3r_encoder import SimpleCut3rEncoder
from llava.utils import rank0_print

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sincos position embedding (from DreamVLA/MAE)
    grid_size: int of the grid height and width
    return: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class StreamVLNModel(LlavaQwenModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(StreamVLNModel, self).__init__(config)
        
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

        self.num_history = getattr(config, 'num_history', None)
        

class StreamVLNForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(Qwen2ForCausalLM, self).__init__(config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None
        config.delay_load = True
        
        self.model = StreamVLNModel(config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.use_world_query = getattr(config, 'use_world_query', True)
        rank0_print(f"World Query Token: {'Enabled' if self.use_world_query else 'Disabled'}")
        
        if self.use_world_query:
            self.num_world_query_tokens = getattr(config, 'num_world_query_tokens', 9)
            self.world_query_decoder_depth = getattr(config, 'world_query_decoder_depth', 2)
            self.wm_loss_weight = getattr(config, 'wm_loss_weight', 0.15)
            self.wm_2d_loss_ratio = getattr(config, 'wm_2d_loss_ratio', 0.5)
            self.wm_use_2d_loss = getattr(config, 'wm_use_2d_loss', True)
            self.wm_use_3d_loss = getattr(config, 'wm_use_3d_loss', True)
            self.wm_2d_loss_type = getattr(config, 'wm_2d_loss_type', 'mse')
            hidden_size = config.hidden_size
            rank0_print(f"hidden_size: {hidden_size}")

            valid_loss_types = ['mse', 'cosine']
            if self.wm_2d_loss_type not in valid_loss_types:
                raise ValueError(
                    f"wm_2d_loss_type must be one of {valid_loss_types}, got {self.wm_2d_loss_type}"
                )
            rank0_print(f"2D Loss Type: {self.wm_2d_loss_type}")
            
            self.world_query_2d = nn.Parameter(torch.zeros(1, self.num_world_query_tokens, hidden_size))
            self.world_query_3d = nn.Parameter(torch.zeros(1, self.num_world_query_tokens, hidden_size))
            nn.init.normal_(self.world_query_2d, std=0.02)
            nn.init.normal_(self.world_query_3d, std=0.02)
            
            self.num_pred_tokens_2d = 196  # 13x13 pooled SigLIP tokens
            self.num_pred_tokens_3d = 196  # 13x13 pooled CUT3R tokens
            
            self.dim_2d = 1152  # SigLIP feature dim
            self.dim_3d = 768   # CUT3R feature dim
            
            rank0_print(
                f"World Model 2D: {self.num_pred_tokens_2d} tokens "
                f"({int(self.num_pred_tokens_2d**0.5)}x{int(self.num_pred_tokens_2d**0.5)})"
            )
            rank0_print(
                f"World Model 3D: {self.num_pred_tokens_3d} tokens "
                f"({int(self.num_pred_tokens_3d**0.5)}x{int(self.num_pred_tokens_3d**0.5)})"
            )
            
            self.world_query_2d_projector = nn.Linear(hidden_size, self.dim_2d)
            self.world_mask_token_2d = nn.Parameter(torch.zeros(1, 1, self.dim_2d))
            nn.init.normal_(self.world_mask_token_2d, std=0.02)
            
            self.world_pos_embed_2d = nn.Parameter(
                torch.zeros(1, self.num_world_query_tokens + self.num_pred_tokens_2d, self.dim_2d),
                requires_grad=False
            )
            self._init_world_pos_embed_2d()
            
            self.world_decoder_2d = nn.Sequential(*[
                Block(self.dim_2d, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(self.world_query_decoder_depth)
            ])
            self.world_decoder_2d_norm = nn.LayerNorm(self.dim_2d)
            self.world_decoder_2d_pred = nn.Linear(self.dim_2d, self.dim_2d)
            
            self.world_query_3d_projector = nn.Linear(hidden_size, self.dim_3d)
            self.world_mask_token_3d = nn.Parameter(torch.zeros(1, 1, self.dim_3d))
            nn.init.normal_(self.world_mask_token_3d, std=0.02)
            
            self.world_pos_embed_3d = nn.Parameter(
                torch.zeros(1, self.num_world_query_tokens + self.num_pred_tokens_3d, self.dim_3d),
                requires_grad=False
            )
            self._init_world_pos_embed_3d()
            
            self.world_decoder_3d = nn.Sequential(*[
                Block(self.dim_3d, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
                for _ in range(self.world_query_decoder_depth)
            ])
            self.world_decoder_3d_norm = nn.LayerNorm(self.dim_3d)
            self.world_decoder_3d_pred = nn.Linear(self.dim_3d, self.dim_3d)
            
            sqrt_tokens = int(self.num_world_query_tokens**0.5)
            assert sqrt_tokens**2 == self.num_world_query_tokens, \
                f"num_world_query_tokens must be a perfect square, got {self.num_world_query_tokens}"
            
            rank0_print(f"World Query: {self.num_world_query_tokens} tokens/round")
            rank0_print(f"Decoder depth: {self.world_query_decoder_depth} Transformer blocks")
            rank0_print(f"Loss weights: wm={self.wm_loss_weight}, 2d_ratio={self.wm_2d_loss_ratio}")
    
        self._init_spatial_fusion()
        
        # Cache for generate() during evaluation
        self.curr_t = []
        self.cache = []

        # Initialize weights and apply final processing
        self.post_init()

    def _init_spatial_fusion(self):
        """Initialize SigLIP + CUT3R cross-attention spatial fusion."""
        self.cut3r_encoder = SimpleCut3rEncoder()
        self.cut3r_encoder.eval()
        for param in self.cut3r_encoder.parameters():
            param.requires_grad = False
        rank0_print("Spatial fusion: SigLIP + frozen CUT3R cross-attention")

        d_clip = 1152
        d_spatial = 768
        d_attn = 1152
        num_heads = 18

        self.clip_norm = nn.LayerNorm(d_clip)
        self.spatial_norm = nn.LayerNorm(d_spatial)
        self.clip_query_proj = nn.Linear(d_clip, d_attn)
        self.spatial_key_proj = nn.Linear(d_spatial, d_attn)
        self.spatial_value_proj = nn.Linear(d_spatial, d_attn)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_attn, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(d_attn)
        self.out_proj = nn.Linear(d_attn, d_clip)
        self.fusion_dropout = nn.Dropout(0.1)

    
    def _init_world_pos_embed_2d(self):
        """Initialize 2D position embedding for world query decoder."""
        query_pos = get_2d_sincos_pos_embed(
            self.dim_2d,
            int(self.num_world_query_tokens**0.5),
            cls_token=False
        )
        mask_pos = get_2d_sincos_pos_embed(
            self.dim_2d,
            int(self.num_pred_tokens_2d**0.5),
            cls_token=False
        )
        pos_embed = np.concatenate([query_pos, mask_pos], axis=0)
        self.world_pos_embed_2d.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def _init_world_pos_embed_3d(self):
        """Initialize 3D position embedding for world query decoder."""
        query_pos = get_2d_sincos_pos_embed(
            self.dim_3d,
            int(self.num_world_query_tokens**0.5),
            cls_token=False
        )
        mask_pos = get_2d_sincos_pos_embed(
            self.dim_3d,
            int(self.num_pred_tokens_3d**0.5),
            cls_token=False
        )
        pos_embed = np.concatenate([query_pos, mask_pos], axis=0)
        self.world_pos_embed_3d.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def get_model(self):
        return self.model
    
    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side # 27
        
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [ceil(height / stride), ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature
    
    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]
        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(3, 4)
        image_feature = torch.cat((image_feature, self.model.image_newline[:,None, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(2, 3).permute(1, 2, 0).contiguous()
        return image_feature
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_rgbd(self, images, depths, poses, intrinsics, time_ids=None, task_ids=None, future_images=None):
        batch_size, num_view, _, H, W = images.shape
        image_features = self.get_model().get_vision_tower()(images.flatten(0, 1))
        
        gt_2d_features = None
        gt_3d_features = None
        
        with torch.no_grad():
            spatial_features = self.cut3r_encoder(images.flatten(0, 1))
        
        clip_features_norm = self.clip_norm(image_features)
        spatial_features_norm = self.spatial_norm(spatial_features)
        
        clip_query = self.clip_query_proj(clip_features_norm)
        spatial_key = self.spatial_key_proj(spatial_features_norm)
        spatial_value = self.spatial_value_proj(spatial_features_norm)
        
        fused_features, _ = self.cross_attention(
            query=clip_query,
            key=spatial_key,
            value=spatial_value
        )
        fused_features = self.out_proj(fused_features)
        fused_features = self.out_norm(fused_features)
        fused_features = fused_features + image_features
        image_features = self.fusion_dropout(fused_features)
        
        if self.use_world_query and future_images is not None and self.training:
            with torch.no_grad():
                gt_2d_features = self.get_model().get_vision_tower()(future_images.flatten(0, 1))
                gt_3d_features = self.cut3r_encoder(future_images.flatten(0, 1))
        
        num_patches_per_side = self.get_model().get_vision_tower().num_patches_per_side
        # (B, V, C, num_patch, num_patch)
        image_features = image_features.permute(0, 2, 1).reshape(batch_size, num_view, -1, num_patches_per_side, num_patches_per_side)
        
        # batch_size, num_view, H, W = depths.shape
        if num_view != 1:
            memory_features = []
            image_features_ = []
            for b in range(batch_size):
                if time_ids[b] is not None:
                    start_idx = time_ids[b][0]
                else:
                    start_idx = 0
                if start_idx == 0:
                    memory_features.append(None)
                    image_features_.append(image_features[b])
                    continue
                else:
                    history_idx = self.model.num_history
                    image_features_.append(image_features[b, history_idx:])
                his_image_feature = image_features[b, :history_idx].flatten(2,3).permute(0,2,1)
                his_image_feature = self.get_model().mm_projector(his_image_feature)
                his_image_feature = self.get_2dPool(his_image_feature, 2) # [N, 196, 1152]
                
                if self.use_world_query and not hasattr(self, '_pooling_verified'):
                    assert his_image_feature.shape[1] == self.num_pred_tokens_2d, \
                        f"get_2dPool output ({his_image_feature.shape[1]}) != world model tokens ({self.num_pred_tokens_2d})"
                    rank0_print(
                        f"Pooling verified: {his_image_feature.shape[1]} tokens "
                        f"({int(his_image_feature.shape[1]**0.5)}x{int(his_image_feature.shape[1]**0.5)})"
                    )
                    self._pooling_verified = True
                
                memory_features.append(his_image_feature.flatten(0,1).unsqueeze(0))
            image_features = image_features_
        else:
            memory_features = [None] * batch_size
        
        image_features_=[]
        for j, image_feature in enumerate(image_features):
            image_feature = image_feature.flatten(2,3).permute(0,2,1)
            image_feature = self.get_model().mm_projector(image_feature)
            image_feature = self.get_2dPool(image_feature, 2)
            image_features_.append(image_feature)
        image_features = image_features_
        
        return image_features, memory_features, gt_2d_features, gt_3d_features
   
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, 
        images, image_sizes, depths, poses, intrinsics, time_ids=None, task_ids=None,
        future_images=None, future_depths=None
    ):  
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        image_features, memory_features, gt_2d_features, gt_3d_features = self.encode_rgbd(
            images, depths, poses, intrinsics, time_ids, task_ids, future_images
        )

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_memories = (cur_input_ids == MEMORY_TOKEN_INDEX).sum()
            # print(batch_idx, num_images, num_memories)
            num_specials = num_images + num_memories
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
            memory_token_indices = torch.where(cur_input_ids == MEMORY_TOKEN_INDEX)[0].tolist()
            special_token_indices = sorted(image_token_indices + memory_token_indices)
            special_tokens = [cur_input_ids[indice] for indice in special_token_indices]
            special_token_indices = [-1] + special_token_indices + [cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[special_token_indices[i]+1:special_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[special_token_indices[i]+1:special_token_indices[i+1]])
                
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            
            cur_img_id = 0
            cur_mem_id = 0
            
            for i in range(num_specials + 1):  # num_images = 1? [0, 1]
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_specials:
                    # print(f"Batch Index: {batch_idx}\n, Current Image Index: {cur_image_idx}\n, Num Images: {num_images}")
                    special_token = special_tokens[i]
                
                    if special_token == IMAGE_TOKEN_INDEX:
                        cur_image_feature = image_features[batch_idx][cur_img_id]
                        cur_img_id += 1
                        # print(batch_idx, i, 'cur_image_feature shape:', cur_image_feature.shape)
                        cur_new_input_embeds.append(cur_image_feature)
                        cur_new_labels.append(torch.full((cur_image_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif special_token == MEMORY_TOKEN_INDEX:
                        cur_memory_feature = memory_features[batch_idx][cur_mem_id]
                        cur_mem_id += 1
                        # print(batch_idx, i, 'cur_memory_feature shape:', cur_memory_feature.shape)
                        cur_new_input_embeds.append(cur_memory_feature)
                        cur_new_labels.append(torch.full((cur_memory_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    else:
                        raise NotImplementedError
            
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            # assert len(cur_new_input_embeds) <= 4096
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
            
        if _attention_mask is None:
            attention_mask = None
        else:
            assert attention_mask.shape[1] == new_input_embeds.shape[1], \
                f"attention_mask length ({attention_mask.shape[1]}) != new_input_embeds ({new_input_embeds.shape[1]})"
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        
        position_ids = None
        
        world_query_data = None
        if self.use_world_query and self.training and future_images is not None and (self.wm_use_2d_loss or self.wm_use_3d_loss):
            batch_size = new_input_embeds.shape[0]
            num_rounds = future_images.shape[1]
            nav_seq_len = new_input_embeds.shape[1]
            
            num_query_per_round = 0
            if self.wm_use_2d_loss:
                num_query_per_round += self.num_world_query_tokens
            if self.wm_use_3d_loss:
                num_query_per_round += self.num_world_query_tokens
            
            round_end_positions = []
            actual_num_rounds_per_sample = []
            
            for b in range(batch_size):
                valid_positions = (new_labels[b] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
                
                if len(valid_positions) == 0:
                    round_end_positions.append([nav_seq_len] * num_rounds)
                    actual_num_rounds_per_sample.append(0)
                    continue
                
                positions = valid_positions.tolist()
                answer_end_positions = []
                
                for i in range(len(positions) - 1):
                    if positions[i+1] != positions[i] + 1:
                        answer_end_positions.append(positions[i] + 1)
                
                if len(positions) > 0:
                    answer_end_positions.append(positions[-1] + 1)
                
                actual_rounds = len(answer_end_positions)
                actual_num_rounds_per_sample.append(min(actual_rounds, num_rounds))
                
                if actual_rounds < num_rounds:
                    last_pos = answer_end_positions[-1] if answer_end_positions else nav_seq_len
                    answer_end_positions.extend([last_pos] * (num_rounds - actual_rounds))
                else:
                    answer_end_positions = answer_end_positions[:num_rounds]
                
                round_end_positions.append(answer_end_positions)
            
            round_end_positions = torch.tensor(round_end_positions, device=new_input_embeds.device, dtype=torch.long)
            actual_num_rounds_per_sample = torch.tensor(actual_num_rounds_per_sample, device=new_input_embeds.device, dtype=torch.long)
            
            query_embeds_list = []
            query_positions_list = []
            
            for b in range(batch_size):
                sample_query_embeds = []
                sample_query_positions = []
                actual_rounds = actual_num_rounds_per_sample[b].item()
                
                current_pos = nav_seq_len
                
                for round_idx in range(actual_rounds):
                    if self.wm_use_2d_loss:
                        sample_query_embeds.append(self.world_query_2d[0])
                        sample_query_positions.append({
                            'round_idx': round_idx,
                            'type': '2d',
                            'start': current_pos,
                            'end': current_pos + self.num_world_query_tokens
                        })
                        current_pos += self.num_world_query_tokens
                    
                    if self.wm_use_3d_loss:
                        sample_query_embeds.append(self.world_query_3d[0])
                        sample_query_positions.append({
                            'round_idx': round_idx,
                            'type': '3d',
                            'start': current_pos,
                            'end': current_pos + self.num_world_query_tokens
                        })
                        current_pos += self.num_world_query_tokens
                
                total_query_tokens = num_rounds * num_query_per_round
                if len(sample_query_embeds) > 0:
                    query_embeds = torch.cat(sample_query_embeds, dim=0)
                    if query_embeds.shape[0] < total_query_tokens:
                        padding = torch.zeros(
                            total_query_tokens - query_embeds.shape[0],
                            query_embeds.shape[1],
                            dtype=query_embeds.dtype,
                            device=query_embeds.device
                        )
                        query_embeds = torch.cat([query_embeds, padding], dim=0)
                else:
                    query_embeds = torch.zeros(
                        total_query_tokens,
                        new_input_embeds.shape[-1],
                        dtype=new_input_embeds.dtype,
                        device=new_input_embeds.device
                    )
                
                query_embeds_list.append(query_embeds)
                query_positions_list.append(sample_query_positions)
            
            query_embeds_batch = torch.stack(query_embeds_list, dim=0)
            new_input_embeds = torch.cat([new_input_embeds, query_embeds_batch], dim=1)
            
            query_labels = torch.full(
                (batch_size, query_embeds_batch.shape[1]),
                IGNORE_INDEX,
                dtype=new_labels.dtype,
                device=new_labels.device
            )
            new_labels = torch.cat([new_labels, query_labels], dim=1)
            
            if attention_mask is not None:
                query_mask = torch.ones(
                    (batch_size, query_embeds_batch.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([attention_mask, query_mask], dim=1)
            
            world_query_data = {
                'gt_2d_features': gt_2d_features,
                'gt_3d_features': gt_3d_features,
                'num_future': num_rounds,
                'round_end_positions': round_end_positions,
                'actual_num_rounds': actual_num_rounds_per_sample,
                'query_positions': query_positions_list,
                'nav_seq_len': nav_seq_len,
                'num_query_per_round': num_query_per_round
            }

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, world_query_data

    def compute_2d_loss(self, pred_features, gt_features, loss_type='mse'):
        """Compute 2D latent prediction loss with mse or cosine."""
        assert pred_features.shape == gt_features.shape, \
            f"pred/gt shape mismatch: pred={pred_features.shape}, gt={gt_features.shape}"
        assert len(pred_features.shape) == 3, \
            f"Expected 3D tensor [B, N, D], got shape={pred_features.shape}"
        assert pred_features.shape[-1] == self.dim_2d, \
            f"Feature dim mismatch: expected {self.dim_2d}, got {pred_features.shape[-1]}"
        
        if loss_type == 'mse':
            return nn.functional.mse_loss(pred_features, gt_features.detach())
        
        if loss_type == 'cosine':
            pred_norm = nn.functional.normalize(pred_features, p=2, dim=-1)
            gt_norm = nn.functional.normalize(gt_features.detach(), p=2, dim=-1)
            cosine_sim = (pred_norm * gt_norm).sum(dim=-1)
            return (1.0 - cosine_sim).mean()
        
        raise ValueError(f"Unknown 2D loss type: {loss_type}. Use 'mse' or 'cosine'.")
    
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
        images: torch.FloatTensor = None,
        depths: torch.FloatTensor = None,
        poses: torch.FloatTensor = None,
        intrinsics: torch.FloatTensor = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        future_images: Optional[torch.FloatTensor] = None,
        future_depths: Optional[torch.FloatTensor] = None,  # kept for dataloader API compatibility
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        tokenizer = kwargs.get("tokenizer", None)
        input_ids_ = input_ids
        time_ids = kwargs.get("time_ids", None)
        task_ids = kwargs.get("task_type", None)
        world_query_data = None
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                world_query_data
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels, 
                images, 
                image_sizes,
                depths, 
                poses, 
                intrinsics,
                time_ids,
                task_ids,
                future_images,
                future_depths
            )
        def create_query_isolation_mask(attention_mask, world_query_data):
            """
            Build 4D attention mask with per-query visibility and cross-round isolation.
            Each query sees navigation tokens up to its round end; same-round same-type
            queries may attend to each other.
            """
            batch_size, seq_len = attention_mask.shape
            device = attention_mask.device
            
            if attention_mask.dtype == torch.bool:
                dtype = torch.bfloat16
            else:
                dtype = attention_mask.dtype
            
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=dtype, device=device))
            mask_4d = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
            
            round_end_positions = world_query_data['round_end_positions']
            query_positions = world_query_data['query_positions']
            
            for b in range(batch_size):
                for query_info in query_positions[b]:
                    round_idx = query_info['round_idx']
                    q_start = query_info['start']
                    q_end = query_info['end']
                    round_end = round_end_positions[b, round_idx].item()
                    
                    mask_4d[b, 0, q_start:q_end, :] = 0
                    mask_4d[b, 0, q_start:q_end, :round_end] = 1
                    mask_4d[b, 0, q_start:q_end, q_start:q_end] = 1
            
            padding_4d = (
                attention_mask[:, None, :, None] *
                attention_mask[:, None, None, :]
            ).to(dtype)
            mask_4d = mask_4d * padding_4d
            
            return (1.0 - mask_4d) * torch.finfo(dtype).min
        
        attention_mask_for_model = attention_mask
        if world_query_data is not None and attention_mask is not None:
            if world_query_data:
                attention_mask_for_model = create_query_isolation_mask(attention_mask, world_query_data)
    
        need_hidden_states = (self.use_world_query and self.training and world_query_data is not None)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask_for_model,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=need_hidden_states,
            return_dict=True
        )
        
        if self.use_world_query and self.training and world_query_data is not None:
            hidden_states = outputs.hidden_states[-1]
            batch_size = hidden_states.shape[0]
            num_rounds = world_query_data['num_future']
            num_query_per_round = world_query_data['num_query_per_round']
            
            wq_2d_hidden_list = []
            wq_3d_hidden_list = []
            valid_sample_indices = []
            
            query_positions = world_query_data['query_positions']
            actual_num_rounds_list = world_query_data['actual_num_rounds']
            
            for b in range(batch_size):
                actual_rounds = actual_num_rounds_list[b].item()
                if actual_rounds == 0:
                    continue
                
                sample_2d_hidden = []
                sample_3d_hidden = []
                
                queries_sorted = sorted(query_positions[b], key=lambda x: (x['round_idx'], x['type']))
                
                for query_info in queries_sorted:
                    q_start = query_info['start']
                    q_end = query_info['end']
                    q_type = query_info['type']
                    
                    query_hidden = hidden_states[b, q_start:q_end]
                    
                    if q_type == '2d':
                        sample_2d_hidden.append(query_hidden)
                    elif q_type == '3d':
                        sample_3d_hidden.append(query_hidden)
                if sample_2d_hidden:
                    wq_2d_hidden_list.append(torch.stack(sample_2d_hidden, dim=0))
                if sample_3d_hidden:
                    wq_3d_hidden_list.append(torch.stack(sample_3d_hidden, dim=0))
                valid_sample_indices.append(b)
            
            if not wq_2d_hidden_list and not wq_3d_hidden_list:
                rank0_print("Warning: no valid world query tokens in current batch")
                if not return_dict:
                    return outputs.to_tuple()
                return outputs
            
            valid_batch_size = len(valid_sample_indices)
            max_rounds = num_rounds
            if wq_2d_hidden_list:
                max_rounds = max(max_rounds, max(t.shape[0] for t in wq_2d_hidden_list))
            if wq_3d_hidden_list:
                max_rounds = max(max_rounds, max(t.shape[0] for t in wq_3d_hidden_list))
            
            if wq_2d_hidden_list: 
                padded_2d_list = []
                for tokens in wq_2d_hidden_list:
                    if tokens.shape[0] < max_rounds:
                        padding = torch.zeros(
                            max_rounds - tokens.shape[0],
                            tokens.shape[1],
                            tokens.shape[2],
                            dtype=tokens.dtype,
                            device=tokens.device
                        )
                        tokens = torch.cat([tokens, padding], dim=0)
                    padded_2d_list.append(tokens)
                wq_2d_hidden = torch.stack(padded_2d_list, dim=0)
                wq_2d_hidden = wq_2d_hidden.view(-1, wq_2d_hidden.shape[2], wq_2d_hidden.shape[3])
            else:
                wq_2d_hidden = None
            
            if wq_3d_hidden_list:
                padded_3d_list = []
                for tokens in wq_3d_hidden_list:
                    if tokens.shape[0] < max_rounds:
                        padding = torch.zeros(
                            max_rounds - tokens.shape[0],
                            tokens.shape[1],
                            tokens.shape[2],
                            dtype=tokens.dtype,
                            device=tokens.device
                        )
                        tokens = torch.cat([tokens, padding], dim=0)
                    padded_3d_list.append(tokens)
                wq_3d_hidden = torch.stack(padded_3d_list, dim=0)
                wq_3d_hidden = wq_3d_hidden.view(-1, wq_3d_hidden.shape[2], wq_3d_hidden.shape[3])
            else:
                wq_3d_hidden = None
            
            pred_2d = None
            if wq_2d_hidden is not None:
                wq_2d_proj = self.world_query_2d_projector(wq_2d_hidden)
                
                assert wq_2d_proj.shape[-1] == self.dim_2d, \
                    f"2D projection dim error: expected {self.dim_2d}, got {wq_2d_proj.shape[-1]}"
                
                # Add mask tokens: [valid_B*max_rounds, 9, 1152] + [valid_B*max_rounds, 196, 1152] → [valid_B*max_rounds, 205, 1152]
                mask_tokens_2d = self.world_mask_token_2d.expand(
                    valid_batch_size * max_rounds, self.num_pred_tokens_2d, -1
                )
                decoder_input_2d = torch.cat([wq_2d_proj, mask_tokens_2d], dim=1)
                
                expected_seq_len = self.num_world_query_tokens + self.num_pred_tokens_2d
                assert decoder_input_2d.shape[1] == expected_seq_len, \
                    f"2D decoder seq len error: expected {expected_seq_len}, got {decoder_input_2d.shape[1]}"
                
                # Add position embedding
                decoder_input_2d = decoder_input_2d + self.world_pos_embed_2d
                
                # Pass through Transformer Blocks
                decoder_output_2d = self.world_decoder_2d(decoder_input_2d)
                
                # Take only the mask token part: [valid_B*8, 205, 1152] → [valid_B*8, 196, 1152]
                pred_features_2d = decoder_output_2d[:, self.num_world_query_tokens:, :]
                pred_features_2d = self.world_decoder_2d_norm(pred_features_2d)
                pred_2d = self.world_decoder_2d_pred(pred_features_2d)
                
                assert pred_2d.shape == (valid_batch_size * max_rounds, self.num_pred_tokens_2d, self.dim_2d), \
                    f"2D pred shape error: expected ({valid_batch_size * max_rounds}, {self.num_pred_tokens_2d}, {self.dim_2d}), got {pred_2d.shape}"
                
                pred_2d = pred_2d.view(valid_batch_size, max_rounds, self.num_pred_tokens_2d, self.dim_2d)
            
            pred_3d = None
            if wq_3d_hidden is not None:
                wq_3d_proj = self.world_query_3d_projector(wq_3d_hidden)
                
                assert wq_3d_proj.shape[-1] == self.dim_3d, \
                    f"3D projection dim error: expected {self.dim_3d}, got {wq_3d_proj.shape[-1]}"
                
                # Add mask tokens
                mask_tokens_3d = self.world_mask_token_3d.expand(
                    valid_batch_size * max_rounds, self.num_pred_tokens_3d, -1
                )
                decoder_input_3d = torch.cat([wq_3d_proj, mask_tokens_3d], dim=1)
                
                expected_seq_len = self.num_world_query_tokens + self.num_pred_tokens_3d
                assert decoder_input_3d.shape[1] == expected_seq_len, \
                    f"3D decoder seq len error: expected {expected_seq_len}, got {decoder_input_3d.shape[1]}"
                
                # Add position embedding
                decoder_input_3d = decoder_input_3d + self.world_pos_embed_3d
                
                # Pass through Transformer Blocks
                decoder_output_3d = self.world_decoder_3d(decoder_input_3d)
                
                # Take only the mask token part
                pred_features_3d = decoder_output_3d[:, self.num_world_query_tokens:, :]
                pred_features_3d = self.world_decoder_3d_norm(pred_features_3d)
                pred_3d = self.world_decoder_3d_pred(pred_features_3d)
                
                assert pred_3d.shape == (valid_batch_size * max_rounds, self.num_pred_tokens_3d, self.dim_3d), \
                    f"3D pred shape error: expected ({valid_batch_size * max_rounds}, {self.num_pred_tokens_3d}, {self.dim_3d}), got {pred_3d.shape}"
                
                pred_3d = pred_3d.view(valid_batch_size, max_rounds, self.num_pred_tokens_3d, self.dim_3d)
            
            device = hidden_states.device
            loss_2d = torch.tensor(0.0, device=device)
            loss_3d = torch.tensor(0.0, device=device)
            
            valid_rounds_mask = torch.zeros(valid_batch_size, max_rounds, device=device, dtype=torch.bool)
            for i, sample_idx in enumerate(valid_sample_indices):
                actual_rounds = actual_num_rounds_list[sample_idx].item()
                valid_rounds_mask[i, :actual_rounds] = True
            
            def select_valid_gt_features(gt_features, valid_indices, num_rounds):
                """Select GT features for valid samples only."""
                valid_gt_indices = [idx * num_rounds + r for idx in valid_indices for r in range(num_rounds)]
                return gt_features[valid_gt_indices]
                
            def pool_gt_features(gt_features, target_tokens, target_dim, valid_batch_size, num_rounds):
                """Pool GT features to target size using adaptive pooling."""
                if gt_features.shape[1] == target_tokens:
                    return gt_features.view(valid_batch_size, num_rounds, -1, target_dim)
                
                batch_future = gt_features.shape[0]
                side = int(np.sqrt(gt_features.shape[1]))
                target_side = int(np.sqrt(target_tokens))
                    
                gt_features = gt_features.view(batch_future, side, side, -1).permute(0, 3, 1, 2)
                gt_features = nn.functional.adaptive_avg_pool2d(gt_features, (target_side, target_side))
                gt_features = gt_features.permute(0, 2, 3, 1).contiguous()
                gt_features = gt_features.view(batch_future, target_side * target_side, target_dim)
                
                return gt_features.view(valid_batch_size, num_rounds, -1, target_dim)

            if self.wm_use_2d_loss and pred_2d is not None and world_query_data['gt_2d_features'] is not None:
                gt_2d = select_valid_gt_features(world_query_data['gt_2d_features'], valid_sample_indices, max_rounds)
                gt_2d = pool_gt_features(gt_2d, self.num_pred_tokens_2d, self.dim_2d, valid_batch_size, max_rounds)
                
                assert pred_2d.shape == gt_2d.shape, \
                    f"2D pred/gt shape mismatch: pred_2d={pred_2d.shape}, gt_2d={gt_2d.shape}"
                assert pred_2d.shape == (valid_batch_size, max_rounds, self.num_pred_tokens_2d, self.dim_2d), \
                    f"2D pred shape error: expected ({valid_batch_size}, {max_rounds}, {self.num_pred_tokens_2d}, {self.dim_2d}), got {pred_2d.shape}"

                mask_expanded = valid_rounds_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_2d)
                valid_pred = pred_2d[mask_expanded].view(-1, self.dim_2d)
                valid_gt = gt_2d[mask_expanded].view(-1, self.dim_2d)
                if valid_pred.numel() > 0:
                    loss_2d = self.compute_2d_loss(
                        valid_pred.unsqueeze(0), valid_gt.unsqueeze(0), loss_type=self.wm_2d_loss_type
                    )
                else:
                    loss_2d = torch.tensor(0.0, device=device)
            
            if self.wm_use_3d_loss and pred_3d is not None and world_query_data['gt_3d_features'] is not None:
                gt_3d = select_valid_gt_features(world_query_data['gt_3d_features'], valid_sample_indices, max_rounds)
                gt_3d = pool_gt_features(gt_3d, self.num_pred_tokens_3d, self.dim_3d, valid_batch_size, max_rounds)
                
                assert pred_3d.shape == gt_3d.shape, \
                    f"3D pred/gt shape mismatch: pred_3d={pred_3d.shape}, gt_3d={gt_3d.shape}"
                assert pred_3d.shape == (valid_batch_size, max_rounds, self.num_pred_tokens_3d, self.dim_3d), \
                    f"3D pred shape error: expected ({valid_batch_size}, {max_rounds}, {self.num_pred_tokens_3d}, {self.dim_3d}), got {pred_3d.shape}"
                
                mask_expanded = valid_rounds_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pred_3d)
                valid_pred = pred_3d[mask_expanded].view(-1)
                valid_gt = gt_3d[mask_expanded].view(-1)
                if valid_pred.numel() > 0:
                    loss_3d = nn.functional.mse_loss(valid_pred, valid_gt.detach())
            
            has_2d_loss = self.wm_use_2d_loss and loss_2d.numel() > 0 and not torch.isnan(loss_2d)
            has_3d_loss = self.wm_use_3d_loss and loss_3d.numel() > 0 and not torch.isnan(loss_3d)
            
            if has_2d_loss and has_3d_loss:
                world_loss = self.wm_2d_loss_ratio * loss_2d + (1.0 - self.wm_2d_loss_ratio) * loss_3d
            elif has_2d_loss:
                world_loss = loss_2d
            elif has_3d_loss:
                world_loss = loss_3d
            else:
                world_loss = torch.tensor(0.0, device=device, dtype=loss_2d.dtype)
            
            if outputs.loss is not None:
                outputs.nav_loss = outputs.loss.clone()
                outputs.loss = outputs.loss + self.wm_loss_weight * world_loss
            else:
                outputs.nav_loss = torch.tensor(0.0, device=world_loss.device)
                outputs.loss = self.wm_loss_weight * world_loss
            
            outputs.wm_loss = world_loss
            outputs.wm_2d_loss = loss_2d
            outputs.wm_3d_loss = loss_3d
        
        if return_dict is None:
            return_dict = self.config.use_return_dict
        
        if not return_dict:
            return outputs.to_tuple()
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        depths: Optional[torch.FloatTensor] = None,
        poses: Optional[torch.FloatTensor] = None,
        intrinsics: Optional[torch.FloatTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        time_ids = kwargs.pop("time_ids", None)
        task_ids = kwargs.pop("task_type", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
                depths,
                poses,
                intrinsics,
                time_ids,
                task_ids,
                None
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        env_id = kwargs.pop("env_id", None)
        if self.curr_t[env_id] == 0:
            self.cache[env_id]["inputs_embeds"] = inputs_embeds
        else:
            self.cache[env_id]["inputs_embeds"] = torch.cat([self.cache[env_id]["inputs_embeds"], inputs_embeds],dim=1)
        self.curr_t[env_id] += 1
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=self.cache[env_id]["inputs_embeds"],
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # print('inputs_embeds', inputs_embeds.shape)
        # print('input_ids', input_ids, cache_position)
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]
        # print('input_ids', input_ids, cache_position, cache_position.shape)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # print('cache_position_prepare:', cache_position, len(cache_position))
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        elif inputs_embeds is not None and len(cache_position) > 1:
            model_inputs = {"inputs_embeds": inputs_embeds[:, -len(cache_position):], "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": None, #position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        if images is not None:
            model_inputs['images'] = images
        if image_sizes is not None:
            model_inputs['image_sizes'] = image_sizes
        return model_inputs
    
    def reset(self, env_num):
        """Reset caches for all environments."""
        self.curr_t = [0] * env_num
        self.cache = [dict() for _ in range(env_num)]
    
    def reset_for_env(self, env_idx):
        """Reset cache for a single environment."""
        self.curr_t[env_idx] = 0
        self.cache[env_idx] = dict()
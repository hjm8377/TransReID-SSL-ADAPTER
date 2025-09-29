import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.modules import MSDeformAttn

import vit_pytorch
from vit_pytorch import TransReID, trunc_normal_
from torch.nn.init import normal_

from .adapter_module import (InteractionBlock, SpatialPriorModule,
                             deform_inputs)

_logger = logging.getLogger(__name__)

class TransReIDAdapter(TransReID):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, *args, **kwargs):
        super().__init__(num_heads=num_heads, *args, **kwargs)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplnes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
                             for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4
    
    def forward_features(self, x, camera_id, view_id):
        B, _, img_H, img_W = x.shape

        # SPM
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # ViT
        x_vit, H, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)

        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x_vit[:, 1:, :] = x_vit[:, 1:, :] + pos_embed

        # SIE 
        if self.cam_num > 0 and self.view_num > 0:
            x_vit = x_vit + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x_vit = x_vit + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x_vit = x_vit + self.sie_xishu * self.sie_embed[view_id]

        x_vit = self.pos_drop(x_vit)

        # Interaction loop
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            
            x_vit, c = layer(x_vit, c, self.blocks[indexes[0]:indexes[-1] + 1],
                             deform_inputs1, deform_inputs2, H, W)
            
        x_vit = self.norm(x_vit)

        if self.gem_pool:
            gf = self.gem(x_vit[:, 1:].permute(0, 2, 1)).squeeze()
            return x_vit[:, 0] + gf
        else:
            return x_vit[:, 0]

    def forward(self, x, cam_label=None, view_label=None):
        # The forward pass now correctly uses camera and view labels
        return self.forward_features(x, cam_label, view_label)


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0,local_feature=False,sie_xishu=1.5, **kwargs):
    model = TransReIDAdapter(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, camera=camera, view=view, drop_path_rate=drop_path_rate, sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)
    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReIDAdapter(img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,drop_path_rate=drop_path_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,  **kwargs)
    model.in_planes = 384
    return model
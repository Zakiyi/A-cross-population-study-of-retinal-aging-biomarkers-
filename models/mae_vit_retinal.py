import torch
import timm
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import VisionTransformer

START_AGE = 0
END_AGE = 77


class CascadeHead(nn.Module):
    def __init__(self, embedding_dim, input_dim, num_class=8, full_class=77, age_bin=1, age_prompt='hard', depth_ca=1):
        super(CascadeHead, self).__init__()

        self.age_bin = age_bin
        self.age_prompt = age_prompt
        print('age prompt mode is {}'.format(age_prompt))

        self.norm_base = nn.Identity()

        self.fc_0 = nn.Sequential(nn.Dropout(0.3), nn.Linear(embedding_dim, num_class))
        self.fc_1 = nn.Linear(num_class, embedding_dim //2)

        self.ca_blocks = Block(dim=embedding_dim, num_heads=6, drop=0.2)
        self.age_prototypes = torch.nn.Parameter(torch.randn(num_class, embedding_dim))

        self.fc_2 = nn.Sequential(nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim), nn.ReLU())
        self.fc_3 = nn.Linear(embedding_dim, full_class)

    def forward(self, x):
        # coarse prediction with class token
        feat_1 = self.norm_base(x[:, 0])    # B C

        out1 = self.fc_0(feat_1)     # coarse logits  B C --> B num_class

        age_span = torch.arange(START_AGE, END_AGE, step=self.age_bin, dtype=torch.float32).to(x.device)
        out1_ = nn.Softmax(dim=-1)(out1) * age_span  # B * class   coarse predictions

        out1_feat = self.fc_1(out1_)

        # pick class with largest prob
        if self.age_prompt == 'hard':
            cls_index = torch.div(out1_.sum(1), self.age_bin, rounding_mode='floor').long().detach()
            prompt_token = self.age_prototypes[cls_index]  # B * dim

        elif self.age_prompt == 'soft':
            prompt_token = torch.matmul(nn.Softmax(dim=-1)(out1).detach(), self.age_prototypes)

        else:
            raise ValueError

        # x = torch.cat((x[:, 0].unsqueeze(1), prompt_token.unsqueeze(1), x[:, 1:]), dim=1)
        x = self.ca_blocks(x)   # B dim

        feat_2 = x[:, 0]
        feat_ = torch.concat([out1_feat, feat_2], dim=1)
        feat_ = self.fc_2(feat_)
        out2 = self.fc_3(feat_)   # final prediction
        return out1, out2, feat_, (feat_1, feat_2)


def load_pretrained_model(model, checkpoint):
    model.load_state_dict(checkpoint, strict=False)
    print('load mae pretrained model!!!')
    return model


class RegModel(VisionTransformer):
    def __init__(self, backbone='vit-small-imagenet', img_size=384, num_class=8, age_bin=10, embedding_dim=384, dropout=0.2):
        super().__init__(img_size=img_size, drop_path_rate=dropout, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
        
        self.age_bin = age_bin
        print('age bin is {}'.format(self.age_bin))

        self.head = CascadeHead(self.embed_dim, self.embed_dim, num_class, age_bin=self.age_bin)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        return x
        
    def forward(self, images):
        x = self.forward_features(images) # b n c
        out1, out2, feat_, feats  = self.head(x)
        return out1, out2, feat_, feats

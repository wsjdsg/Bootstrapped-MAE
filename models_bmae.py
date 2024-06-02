# implementation of bootstrappted mae
# main idea: maintain a shadow model
# 1.w/o EMA
#   * 1.1 first args.epochs/args.bootstrap_k epochs train the same as mae
#   * 1.2 then the labels should be the output of the encoder of the shadow model
# 2.w/ EMA
#   * 2.1 first K epochs train the same as mae,K as a hyper parameter
#   * 2.2 then each epoch we update the shadow model and the label is the output of encoder of shadow model
# 3.the last proj layer of decoder should be changed when needed,
#       i.e. Linear(decoder_embed,p*p*channel)-> Linear(decoder_embed,encoder_embed)

from functools import partial

import torch
import torch.nn as nn
import copy
from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from models_mae import MaskedAutoencoderViT

class BootstrappedMaskedAutoencoderViT(MaskedAutoencoderViT):
    def __init__(self,**kwargs):
        super(BootstrappedMaskedAutoencoderViT,self).__init__(**kwargs)
        
        #Linear(decoder_embed,encoder_embed)
        self.decoder_last_proj = nn.Linear(kwargs['decoder_embed_dim'], kwargs['embed_dim'], bias=True)
        torch.nn.init.xavier_normal_(self.decoder_last_proj.weight)
        if self.decoder_last_proj.bias is not None:
            nn.init.constant_(self.decoder_last_proj.bias, 0)

        self.shadow = {} #shadow model
        #process ema
        assert 'enable_ema' in kwargs.keys()
        self.enable_ema = kwargs['enable_ema']
        if self.enable_ema: 
            self.ema_warmup_epochs = kwargs['ema_warmup_epochs']
            self.ema_register()
            self.ema_alpha = kwargs['ema_alpha']
            self.now_epoch = 0
        
    #TODO
    def ema_register(self):
        # for convenience, I register all the weights including encoder and decoder
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def ema_update(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.ema_alpha) * param.data + self.ema_alpha * self.shadow[name]
                self.shadow[name] = new_average.clone()


    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def bmae_decoder_forward(self,x,ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        #simply replace the last linear layer
        x = self.decoder_last_proj(x)

        x = x[:, 1:, :]
        return x        
        
    def bmae_shadow_encoder_forward(self,x): #don't need to mask
        x = self.shadow['patch_embed'](x) 
        x = x + self.shadow['pos_embed'][:, 1:, :]
        cls_token = self.shadow['cls_token'] + self.shadow['pos_embed'][:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.shadow['blocks']:
            x = blk(x)
        x = self.shadow['norm'](x)

        return x
    
    #TODO normalized
    def bmae_encoder_forward_loss(self, target, pred, mask):
        """
        target: [N, L, embed_dim]
        pred: [N, L , embed_dim]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self,imgs,mask_ratio=0.75):
        if self.enable_ema:
            self.ema_update()
        if (self.enable_ema and self.now_epoch>=self.ema_warmup_epochs) or \
                    (not self.enable_ema and self.shadow):
            # reconstruct encoder
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
            pred = self.bmae_decoder_forward(latent,ids_restore)
            shadow_encoder_out = self.bmae_shadow_encoder_forward(imgs)
            loss = self.bmae_encoder_forward_loss(shadow_encoder_out,pred,mask)
            return loss, pred, mask
        else:
            return super(BootstrappedMaskedAutoencoderViT,self).forward(imgs,mask_ratio)

    def update_shadow(self): #BMAE_K
        self.shadow['patch_embed'] = copy.deepcopy(self.patch_embed)
        self.shadow['cls_token'] = copy.deepcopy(self.cls_token)
        self.shadow['pos_embed'] = copy.deepcopy(self.pos_embed)
        self.shadow['blocks'] = copy.deepcopy(self.blocks)
        self.shadow['norm'] = copy.deepcopy(self.norm)
 
#fixme: I change decoder to a smaller one
#"the decoder can be narrower and shallower"
def deit_tiny(**kwargs):
    model = MaskedAutoencoderViT(
        img_size=32,patch_size=4, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=192, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-12), **kwargs)
    return model

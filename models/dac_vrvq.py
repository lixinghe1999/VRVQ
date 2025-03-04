import math
from typing import List, Union
import numpy as np
import torch
from torch import nn

from audiotools import AudioSignal
from audiotools import AudioSignal
from audiotools.ml import BaseModel

from .layers import *

from .dac_base import CodecMixin
from .quantize import ResidualVectorQuantize, VBRResidualVectorQuantize


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int=64,
        strides: List[int]=[2, 4, 8, 8],
        latent_dim: int=512,
    ):
        super().__init__()
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, latent_dim, kernel_size=3, padding=1),
        ]
        
        self.block = nn.Sequential(*self.block)
    
    def forward(self, x, return_feat=False):
        num_blocks = len(self.block)
        for i, layer in enumerate(self.block):
            x = layer(x)
            if i == num_blocks - 3 and return_feat:
                feat = x
        out = x
        if return_feat:
            return out, feat
        return out
    
    
class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class DAC_VRVQ(BaseModel, CodecMixin):
    def __init__(
        self,
        ## Original DAC Configs
        encoder_dim: int = 64, 
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: Union[int, list] = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,  ## quantizer dropout in original paper
        sample_rate: int = 44100,
        
        ## VBR Configs
        model_type: str="VBR", ## in ["VBR", "CBR"]
        full_codebook_rate: float=0.0,  ## rate of samples to use full number of codebooks
        level_min: float=None, ## minimum Scale factor
        level_max: float=None, ## maximum Scale factor
        level_dist: str="uniform", ## in ["uniform", "loguniform"]
        
        detach_imp_map_input: bool = False,
        imp2mask_alpha: float = 1.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        
        self.model_type = model_type
        
        ## Other configs
        # self.full_codebook_rate = full_codebook_rate
        # self.use_framewise_dropout = use_framewise_dropout
        # self.level_min = level_min
        # self.level_max = level_max
        # self.level_dist = level_dist
        # self.operator_mode = operator_mode
        # self.imp2mask_alpha = imp2mask_alpha
        # self.imp2mask_func = imp2mask_func
        
        
        if model_type == "CBR":
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        elif model_type == "VBR":
            self.quantizer = VBRResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
                ### VBR Configs
                full_codebook_rate=full_codebook_rate,
                level_min=level_min,
                level_max=level_max,
                level_dist=level_dist,
                detach_imp_map_input=detach_imp_map_input,
                imp2mask_alpha=imp2mask_alpha,
            )
        else:
            raise ValueError(f"Invalid RVQ model_type: {model_type}")
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    
    def encode(
        self,
        audio_data: torch.Tensor,
        n_quantizers: int = None,
        level: int = 1, ## Scale Factor, only used in VBR inference. 
    ):
        """
        audio_data: (B, 1, T)
        n_quantizers: 
            - Number of quantizers to use.
            - CBR mode if not None.
            
        level:
            - Scale factor for scaling the importance map.
            - VBR mode if not None.
        
        Returns
        =======
        "z": (B, D, T)
            - Quantized continuous representation of input
            - summed
        "codes" : (B, N_q, T)
            - Codebook indices for each codebook
        "latents" : (B, N_q*D, T)
            - Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : (1)
        "vq/codebook_loss" : (1)
        
        """
        # import pdb; pdb.set_trace()
        z, feat = self.encoder(audio_data, return_feat=True)
        if self.model_type == "CBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers,
                         "feat_enc": feat, "level": level}
        
        out_quant_dict = self.quantizer(**quant_inp)
        return out_quant_dict
        # return z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp
    
    def decode(self, z: torch.Tensor):
        """
        z: (B, D, T)
            - Quantized continuous representation of input
        
        """
        return self.decoder(z)
    
    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        level: int = 1,
    ):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        if self.model_type == "CBR":
            enc_inp = {"audio_data": audio_data, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            enc_inp = {"audio_data": audio_data, "n_quantizers": n_quantizers, "level": level}
        out_enc_dict = self.encode(**enc_inp)
        z_q = out_enc_dict["z_q"]
        # z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp = \
        #     self.encode(audio_data, n_quantizers, level)
            
        x = self.decode(z_q)
        
        out_forward_dict = {
            "audio": x[..., :length],
            "z": z_q,
            "codes": out_enc_dict["codes"],
            "latents": out_enc_dict["latents"],
            "vq/commitment_loss": out_enc_dict["commitment_loss"],
            "vq/codebook_loss": out_enc_dict["codebook_loss"],
            "imp_map": out_enc_dict.get("imp_map", None), ## Can be None in CBR model
            "mask_imp": out_enc_dict.get("mask_imp", None), ## Can be None in CBR mode in VBR model
        }
        
        return out_forward_dict
        
        # return {
        #     "audio": x[..., :length],
        #     "z": z,
        #     "codes": codes,
        #     "latents": latents,
        #     "vq/commitment_loss": commitment_loss,
        #     "vq/codebook_loss": codebook_loss,
        #     "imp_map": imp_map,
        #     "mask_imp": mask_imp, ## can be none in CBR mode
        # }
        
        

class EncoderWithFeatureDenoiser(nn.Module):
    """
    """
    def __init__(
        self,
        d_model: int=64,
        strides: List[int]=[2, 4, 8, 8],
        latent_dim: int=512,
        denoise_block_idx = [1, 3], ## Denoising blocks after these blocks.
        clean_train: bool = False,
        feature_denoise_mode: str = "additive",
    ):
        """
        pre-trained model must be loaded.
        SUPER HARD-CODED
        """
        # import pdb; pdb.set_trace()
        super().__init__()
        # assert feature_denoise_mode in ["additive", "masking"]
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        
        d_model_in = d_model
        d_model_list = []
        for idx, stride in enumerate(strides):
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]
            # if idx in denoise_block_idx:
            d_model_list.append(d_model)
            
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, latent_dim, kernel_size=3, padding=1),
        ]
        
        self.block = nn.Sequential(*self.block)
        self.feature_denoise_mode = feature_denoise_mode
        
        ## self.block: 1 + [strides] + 2 = 3 + len(strides)
        ## Here, len(self.block) = 7
    
        ### Additional Denoising Block
        self.strides = strides
        self.dn_block_idx = denoise_block_idx
        # self.block_denoise = []
        # for idx, d_model in enumerate(d_model_list):
            # self.block_denoise +=[DenoisingBlock(d_model)]
            # self.block_denoise +=[DenoisingBlockTCN(d_model)]
        # self.block_denoise = nn.ModuleDict(
        #     {f"denoise_{idx}": block for idx, block in zip(denoise_block_idx, self.block_denoise)}
        # )
        
        if self.feature_denoise_mode in ["additive", "masking", "masking_bn"]:
            self.block_denoise1 = DenoisingBlockTCN(dim_in=d_model_in, dim_out=d_model_list[1], 
                                                    strides_list=strides[:2])
            self.block_denoise2 = DenoisingBlockTCN(dim_in=d_model_list[1], dim_out=d_model_list[3],
                                                    strides_list=strides[2:])
        elif self.feature_denoise_mode == "masking2":
            # self.block_denoise_list = nn.ModuleList(
            #     [DenoisingBlockTCN2(dim_in=d_model_list[i], 
            #                        dim_out=d_model_list[i], 
            #                        R=4,) 
            #      for i in range(len(d_model_list)) if i in denoise_block_idx]
            # )
            # print("d_model_list", d_model_list)
            # print("denoise_block_idx", denoise_block_idx)
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": DenoisingBlockTCN2(dim_in=d_model_list[idx-1], 
                                                       dim_out=d_model_list[idx-1], 
                                                       R=1,) 
                 for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": DenoisingMambaBlock(
                    n_layer=2,
                    in_channels=d_model_list[idx-1],
                    d_state=16,
                    d_conv=4,
                    expand=4,
                    activation='lsigmoid'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_add":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": DenoisingMambaBlock(
                    n_layer=2,
                    in_channels=d_model_list[idx-1],
                    d_state=16,
                    d_conv=4,
                    expand=4,
                    activation='none'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_add_4":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            assert len(denoise_block_idx) == 1
            assert denoise_block_idx[0] == 4
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": DenoisingMambaBlock(
                    n_layer=10,
                    in_channels=d_model_list[idx-1],
                    d_state=16,
                    d_conv=4,
                    expand=4,
                    activation='none'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_mask_4":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            assert len(denoise_block_idx) == 1
            assert denoise_block_idx[0] == 4
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": DenoisingMambaBlock(
                    n_layer=10,
                    in_channels=d_model_list[idx-1],
                    d_state=16,
                    d_conv=4,
                    expand=4,
                    activation='lsigmoid'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=2,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1],
                    output_dim=d_model_list[idx-1],
                    activation='lsigmoid'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED_add":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=2,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1],
                    output_dim=d_model_list[idx-1],
                    activation='none'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED_add_4":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            assert len(denoise_block_idx) == 1
            assert denoise_block_idx[0] == 4
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=2,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1],
                    output_dim=d_model_list[idx-1],
                    activation='none'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED_gelu":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=2,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1],
                    output_dim=d_model_list[idx-1],
                    activation='gelu'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED_mask_4":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            assert len(denoise_block_idx) == 1
            assert denoise_block_idx[0] == 4
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=24,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1]//2,
                    output_dim=d_model_list[idx-1],
                    activation='lsigmoid'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        elif self.feature_denoise_mode == "masking_mamba_ED_add_4":
            from .layers_mamba import DenoisingMambaBlock, FeatureSEMamba
            assert len(denoise_block_idx) == 1
            assert denoise_block_idx[0] == 4
            self.block_denoise_dict = nn.ModuleDict(
                {f"block_denoise_{idx}": FeatureSEMamba(
                    n_blocks=10,
                    input_dim=d_model_list[idx-1],
                    hidden_dim=d_model_list[idx-1]//2,
                    output_dim=d_model_list[idx-1],
                    activation='none'
                ) for idx in denoise_block_idx}
            )
            self.forward_gt = self.forward_gt2
            self.forward_noisy = self.forward_noisy2
        else:
            raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
        
        self.clean_train = clean_train
        if not clean_train:
            self.freeze_non_denoising_blocks()
        
        self.n_blocks = len(self.block) ## 7 
        self.n_strides = len(self.strides) ## 4
        
        assert self.n_blocks == 7
        assert self.n_strides == 4 ## Same with DAC configs. 
        
        self.denoise_block_idx = denoise_block_idx
        # self.check_grads()
        # assert False
        if self.feature_denoise_mode == "masking_bn":
            self.bn1 = nn.BatchNorm1d(d_model_list[1], affine=False)
            self.bn2 = nn.BatchNorm1d(d_model_list[3], affine=False)
        
        """
        blocks:
        0: WNConv1d(1, dm, kernel_size=(7,), stride=(1,), padding=(3,))
        
        
        1: EncoderBlock : dm -> dm*2
        2: EncoderBlock : dm*2 -> dm*4
        
        3: EncoderBlock : dm*4 -> dm*8
        4: EncoderBlock : dm*8 -> dm*16
        
        5: Snake1d : dm*16 -> dm*16
        6:
        
        """
        
    def freeze_non_denoising_blocks(self):
        # for name, param in self.named_parameters():
        #     if "block_denoise" not in name and \
        #         "block_denoise_list" not in name and \
        #             "block_denoise_dict" not in name:
        #         param.requires_grad = False
        for name, param in self.named_parameters():
            if "block_denoise" not in name:
                param.requires_grad = False
                
    def check_grads(self):
        for name, param in self.named_parameters():
            print(name, param.requires_grad)
        
    # @torch.no_grad()
    def forward_gt(self, x_gt):
        with torch.no_grad():
            fmap_gt = {}
            x = self.block[0](x_gt)
            for i in range(1, self.n_blocks):
                x = self.block[i](x).detach()
                if i in [2, 4]:
                    fmap_gt[f"gt_{i}"] = x.detach()
                if i == self.n_strides:
                    fmap_gt["imp_map_input"] = x.detach()
                if i > self.n_strides:
                    break
            return x.detach(), fmap_gt
    
    def forward_noisy(self, x_noisy):
        # freeze = not self.clean_train
        # if freeze:
        #     fmap_noisy = {}
        #     with torch.no_grad():
        #         x = self.block[0](x_noisy)
        #         # x = x.detach()
        #     ## EncoderBlocks: 1, 2, 3, 4
        #     f_dn1 = self.block_denoise1(x)
        #     with torch.no_grad():
        #         x1 = self.block[1](x)
        #         x2 = self.block[2](x1).detach()
        #     if self.feature_denoise_mode == "additive":
        #         x2_out = x2 + f_dn1
        #     elif self.feature_denoise_mode == "masking":
        #         x2_out = x2 * torch.sigmoid(f_dn1)
        #     elif self.feature_denoise_mode == "masking_bn":
        #         x2_out = x2 * torch.sigmoid(f_dn1)
        #         x2_out = self.bn1(x2_out)
        #     else:
        #         raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
        #     fmap_noisy["denoised_2"] = x2_out
            
        #     f_dn2 = self.block_denoise2(x2_out)
        #     with torch.no_grad():
        #         x3 = self.block[3](x2_out)
        #         x4 = self.block[4](x3).detach()
        #     if self.feature_denoise_mode == "additive":
        #         x4_out = x4 + f_dn2
        #     elif self.feature_denoise_mode == "masking":
        #         x4_out = x4 * torch.sigmoid(f_dn2)
        #     elif self.feature_denoise_mode == "masking_bn":
        #         x4_out = x4 * torch.sigmoid(f_dn2)
        #         x4_out = self.bn2(x4_out)
        #     else:
        #         raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
        #     fmap_noisy["denoised_4"] = x4_out
        #     fmap_noisy["imp_map_input"] = x4_out
            
        #     with torch.no_grad():
        #         x = self.block[5](x4_out)
        #         x = self.block[6](x)

        #     return x, fmap_noisy
        # else:
        fmap_noisy = {}
        x = self.block[0](x_noisy)
        ## EncoderBlocks: 1, 2, 3, 4
        # import pdb; pdb.set_trace()
        f_dn1 = self.block_denoise1(x) ## 32, 256, 1920
        x1 = self.block[1](x)
        x2 = self.block[2](x1) ## 32, 256, 1920
        if self.feature_denoise_mode == "additive":
            x2_out = x2 + f_dn1
        elif self.feature_denoise_mode == "masking":
            x2_out = x2 * torch.sigmoid(f_dn1)
        elif self.feature_denoise_mode == "masking_bn":
            x2_out = x2 * torch.sigmoid(f_dn1)
            x2_out = self.bn1(x2_out)
        else:
            raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
        fmap_noisy["denoised_2"] = x2_out
        
        f_dn2 = self.block_denoise2(x2_out)
        x3 = self.block[3](x2_out)
        x4 = self.block[4](x3)
        if self.feature_denoise_mode == "additive":
            x4_out = x4 + f_dn2
        elif self.feature_denoise_mode == "masking":
            x4_out = x4 * torch.sigmoid(f_dn2)
        elif self.feature_denoise_mode == "masking_bn":
            x4_out = x4 * torch.sigmoid(f_dn2)
            x4_out = self.bn2(x4_out)
        else:
            raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
        fmap_noisy["denoised_4"] = x4_out
        fmap_noisy["imp_map_input"] = x4_out
        
        x = self.block[5](x4_out)
        x = self.block[6](x)

        return x, fmap_noisy   
        
    def forward_gt2(self, x_gt):
        with torch.no_grad():
            fmap_gt = {}
            x = self.block[0](x_gt)
            for i in range(1, self.n_blocks):
                x = self.block[i](x).detach()
                if i in range(1, self.n_strides+1):
                    if i in self.denoise_block_idx:
                        fmap_gt[f"gt_{i}"] = x.detach()
                    # fmap_gt[f"gt_{i}"] = x.detach()
                if i == self.n_strides:
                    fmap_gt["imp_map_input"] = x.detach()
                if i > self.n_strides:
                    break
            return x.detach(), fmap_gt
    
    
    def forward_noisy2(self, x_noisy):
        fmap_noisy = {}
        x = self.block[0](x_noisy)
            # x = x.detach()
        assert self.n_strides == 4
        for ii in range(1, self.n_strides+1): ## 1, 2, 3, 4
            x = self.block[ii](x) ## freezed => not updated. 
            if ii in self.denoise_block_idx:
                f_dn = self.block_denoise_dict[f"block_denoise_{ii}"](x)
                if self.feature_denoise_mode == "additive":
                    assert False
                    x = x + f_dn
                elif self.feature_denoise_mode == "masking2":
                    x = x * torch.sigmoid(f_dn)
                elif self.feature_denoise_mode in ["masking_mamba", 
                                                   "masking_mamba_ED", 
                                                   "masking_mamba_ED_gelu",
                                                   "masking_mamba_mask_4",
                                                   "masking_mamba_ED_mask_4",]:
                    x = f_dn * x
                elif self.feature_denoise_mode in ["masking_mamba_ED_add", 
                                                   "masking_mamba_add", 
                                                   "masking_mamba_add_4",
                                                   "masking_mamba_ED_add_4",]:
                    x = x + f_dn
                
                else:
                    raise ValueError(f"Invalid feature_denoise_mode: {self.feature_denoise_mode}")
                fmap_noisy[f"denoised_{ii}"] = x
        
        fmap_noisy["imp_map_input"] = x
        for i in range(self.n_strides+1, self.n_blocks): ## 5, 6
            x = self.block[i](x)
        return x, fmap_noisy
                    
        
    def forward(self, x_noisy, x_gt):
        assert x_noisy is not None
        
        fmaps = {}
        if x_gt is not None:
            x_gt, fmap_gt = self.forward_gt(x_gt)
        else:
            x_gt, fmap_gt = None, None
        
        x_n, fmap_n = self.forward_noisy(x_noisy)
        
        fmaps["gt"] = fmap_gt
        fmaps["noisy"] = fmap_n
        # print("self.denoise_block_idx", self.denoise_block_idx)
        return x_n, fmaps    
    

class DAC_VRVQ_FeatureDenoise(BaseModel, CodecMixin):
    def __init__(
        self,
        ## Original DAC Configs
        encoder_dim: int = 64, 
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        n_codebooks: int = 9,
        codebook_size: Union[int, list] = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,  ## quantizer dropout in original paper
        sample_rate: int = 44100,
        
        ## Feature Denoiser
        denoise_block_idx: List[int] = [1, 3], ## Denoising blocks after these blocks.
        
        ## VBR Configs
        model_type: str="VBR", ## in ["VBR", "CBR"]
        full_codebook_rate: float=0.0,  ## rate of samples to use full number of codebooks
        use_framewise_dropout: bool=False, ## Apply random quantizer dropout to each frame
        level_min: float=None, ## minimum Scale factor
        level_max: float=None, ## maximum Scale factor
        level_dist: str="uniform", ## in ["uniform", "loguniform"]
        operator_mode: str = "scaling", ## in ["scaling", "exponential", "transformed_scaling"] ## Paper: scaling
        
        imp_map_input: str = "feature",
        detach_imp_map_input: bool = False,
        imp2mask_alpha: float = 1.0,
        imp2mask_func: str="logcosh", ## logcosh, square, sigmoid
        clean_train: bool = False,
        feature_denoise_mode: str = "additive", ## in ["additive", "masking"]
        
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)
        # self.encoder = Encoder(encoder_dim, encoder_rates, latent_dim)
        self.denoise_block_idx = denoise_block_idx
        self.feature_denoise_mode = feature_denoise_mode
        self.encoder = EncoderWithFeatureDenoiser(
            d_model=encoder_dim,
            strides=encoder_rates,
            latent_dim=latent_dim,
            denoise_block_idx=denoise_block_idx,
            clean_train=clean_train,
            feature_denoise_mode=feature_denoise_mode,
        )

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.imp_map_input = imp_map_input
        
        self.model_type = model_type
        if model_type == "CBR":
            self.quantizer = ResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
            )
        elif model_type == "VBR":
            self.quantizer = VBRResidualVectorQuantize(
                input_dim=latent_dim,
                n_codebooks=n_codebooks,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                quantizer_dropout=quantizer_dropout,
                ### VBR Configs
                full_codebook_rate=full_codebook_rate,
                use_framewise_masking=use_framewise_dropout,
                level_min=level_min,
                level_max=level_max,
                level_dist=level_dist,
                operator_mode=operator_mode,
                imp_map_input=imp_map_input,
                detach_imp_map_input=detach_imp_map_input,
                imp2mask_alpha=imp2mask_alpha,
                imp2mask_func=imp2mask_func,
            )
        else:
            raise ValueError(f"Invalid RVQ model_type: {model_type}")
        
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
        )
        self.sample_rate = sample_rate
        self.apply(init_weights)
        self.delay = self.get_delay()
        
    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data
    
    def encode(
        self,
        audio_data_noisy: torch.Tensor,
        audio_data_gt: torch.Tensor,
        n_quantizers: int = None,
        level: int = 1, ## Scale Factor, only used in VBR inference. 
    ):
        """
        audio_data: (B, 1, T)
        n_quantizers: 
            - Number of quantizers to use.
            - CBR mode if not None.
            
        level:
            - Scale factor for scaling the importance map.
            - VBR mode if not None.
        
        Returns
        =======
        "z": (B, D, T)
            - Quantized continuous representation of input
            - summed
        "codes" : (B, N_q, T)
            - Codebook indices for each codebook
        "latents" : (B, N_q*D, T)
            - Projected latents (continuous representation of input before quantization)
        "vq/commitment_loss" : (1)
        "vq/codebook_loss" : (1)
        
        """
        # import pdb; pdb.set_trace()        
        z, fmaps = self.encoder(
            x_noisy=audio_data_noisy,
            x_gt=audio_data_gt,
        ) ## fmaps: {"gt": fmap_gt, "noisy": fmap_n}
        
        if self.model_type == "CBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers,
                         "feat_enc": fmaps['noisy']['imp_map_input'], 
                         "level": level}
        
        out_quant_dict = self.quantizer(**quant_inp)
        out_quant_dict["enc_fmaps"] = fmaps
        return out_quant_dict
        # return z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp
    
    def decode(self, z: torch.Tensor):
        """
        z: (B, D, T)
            - Quantized continuous representation of input
        """
        return self.decoder(z)
    
    def forward(
        self,
        audio_data_noisy: torch.Tensor,
        audio_data_clean: torch.Tensor,
        sample_rate: int = None,
        n_quantizers: int = None,
        level: int = 1,
    ):
        length = audio_data_noisy.shape[-1]
        audio_data_noisy = self.preprocess(audio_data_noisy, sample_rate)
        if audio_data_clean is not None:
            audio_data_clean = self.preprocess(audio_data_clean, sample_rate)
        
        enc_inp = {
            "audio_data_noisy": audio_data_noisy,
            "audio_data_gt": audio_data_clean,
            "n_quantizers": n_quantizers,
        }
        if self.model_type == "CBR":
            pass
        elif self.model_type == "VBR":
            enc_inp["level"] = level
            
        out_enc_dict = self.encode(**enc_inp)
        z_q = out_enc_dict["z_q"]
        # z, codes, latents, commitment_loss, codebook_loss, imp_map, mask_imp = \
        #     self.encode(audio_data, n_quantizers, level)
            
        x = self.decode(z_q)
        
        out_forward_dict = {
            "audio": x[..., :length],
            "z": z_q,
            "codes": out_enc_dict["codes"],
            "latents": out_enc_dict["latents"],
            "vq/commitment_loss": out_enc_dict["commitment_loss"],
            "vq/codebook_loss": out_enc_dict["codebook_loss"],
            "imp_map": out_enc_dict.get("imp_map", None), ## Can be None in CBR model
            "mask_imp": out_enc_dict.get("mask_imp", None), ## Can be None in CBR mode in VBR model
        }
        ## update fmaps
        # fmaps_gt = out_enc_dict["gt"]
        # fmaps_noisy = out_enc_dict["noisy"]
        # out_forward_dict.update(out_enc_dict) 
        out_forward_dict["enc_fmaps"] = out_enc_dict["enc_fmaps"]
        
        return out_forward_dict
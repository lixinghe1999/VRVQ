"""
This code is a modified version of the `dac` module from the DAC GitHub repository.  
Original source: https://github.com/descriptinc/descript-audio-codec/blob/main/dac/model/dac.py
"""
import math
from typing import List, Union
import numpy as np
import torch
from torch import nn

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
        z, feat = self.encoder(audio_data, return_feat=True)
        if self.model_type == "CBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers}
        elif self.model_type == "VBR":
            quant_inp = {"z": z, "n_quantizers": n_quantizers,
                         "feat_enc": feat, "level": level}
        
        out_quant_dict = self.quantizer(**quant_inp)
        return out_quant_dict
    
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
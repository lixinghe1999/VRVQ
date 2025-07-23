import math
from typing import List, Union
import numpy as np
import torch
from torch import nn

from audiotools.ml import BaseModel

from .layers import *

from .dac_base import CodecMixin
from .quantize import ResidualVectorQuantize
from .dac_vrvq import Encoder, Decoder
from .utils import generate_mask_ste, generate_mask_hard, generate_mask_ste_moe
from .importance_subnet import ImportanceSubnet


class MOEResidualVectorQuantize(ResidualVectorQuantize):
    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
        ### VBR specific parameters
        full_codebook_rate: float = 0.5,
        level_min: float = 1,
        level_max: float = 1,
        level_dist: str = "uniform", ## in ["uniform", "log_uniform"]
        detach_imp_map_input: bool = False, 
        imp2mask_alpha: float = 1.0,
    ):
        super().__init__(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.full_codebook_rate = full_codebook_rate
        self.level_min = level_min
        self.level_max = level_max
        self.level_dist = level_dist
        self.detach_imp_map_input = detach_imp_map_input
        self.imp2mask_alpha = imp2mask_alpha
        
        self.router = nn.Linear(input_dim, n_codebooks)
    
    def forward(
        self,
        z: torch.Tensor,
        n_quantizers: int = None,
        feat_enc: torch.Tensor = None,
        level: float = 1, ## only used in VBR inference.
    ):
        z_q = 0
        residual = z
        bs, ch, frames = z.shape # (B, D, T)
        
        commitment_loss = torch.zeros(bs, self.n_codebooks, frames).to(z.device)
        codebook_loss = torch.zeros(bs, self.n_codebooks, frames).to(z.device)
        
        codebook_indices = []
        latents = []
        z_q_is = []
        
        if n_quantizers is None:
            mode = "VBR"
            assert level is not None, "level must be specified in VBR mode"
        else:
            mode = "CBR"
            # assert level is None, "level must be None in CBR mode"
        
        for i, quantizer in enumerate(self.quantizers):
            if mode == "CBR" and n_quantizers is not None:
                if i >= n_quantizers:
                    break
                
            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual, loss_per_frame=True)
            z_q_is.append(z_q_i)
            residual = residual - z_q_i ## We do not have to consider the effect of the masking for dropouts: 1. its frame-wise-based, and 2. once z_q_i is masked, then we don't use this residual anymore.
            commitment_loss[:, i, :] = commitment_loss_i
            codebook_loss[:, i, :] = codebook_loss_i
            
            codebook_indices.append(indices_i)
            latents.append(z_e_i)
        
        ## Importance Map
        # z_q_is: [(B, D, T), (B, D, T), ...]
        if mode=="VBR":
            z_q_is_cat = torch.cat(z_q_is, dim=1) # (B, D*N, T)
            
            # feat_enc: (B, D, T) -> (B, T, D)
            feat_enc = feat_enc.permute(0, 2, 1)
            imp_map = self.router(feat_enc) # (B, T, Nq)
            imp_map = imp_map.permute(0, 2, 1)
            
            if self.training:
                assert self.level_min <= self.level_max
                # assert self.level_max < 20, "Level_max is too high, we also multiply n_codebooks when we use Simple Scaling function"
                if self.level_dist == "uniform":
                    random_levels = torch.rand((bs, 1, 1)) * (self.level_max - self.level_min) + self.level_min
                    random_levels = random_levels.to(z)
                elif self.level_dist == "log_uniform":
                    random_levels = torch.rand((bs, 1, 1)) * (math.log(self.level_max) - math.log(self.level_min)) + math.log(self.level_min) ## log uniform
                    random_levels = torch.exp(random_levels).to(z)
                else:
                    raise ValueError("Invalid level_dist")

                imp_map_scaled = imp_map * random_levels * self.n_codebooks
            else: 
                imp_map_scaled = imp_map * level * self.n_codebooks

            mask_imp = generate_mask_ste_moe(
                imp_map_scaled,
                self.n_codebooks,
                alpha=self.imp2mask_alpha,
            ) ## mask_imp: (B, Nq, T)
            print(mask_imp)
        
        elif mode == "CBR":
            imp_map_scaled = torch.ones((bs, 1, frames)).to(z) * n_quantizers
            imp_map = None
            mask_imp = torch.ones((bs, self.n_codebooks, frames)).to(z)
        else:
            raise ValueError("Invalid mode")
        
        ## Dropout / Full Codebook
        if self.training:
            dropout = torch.randint(1, self.n_codebooks + 1, (bs, 1, 1))
            dropout = dropout.expand(bs, 1, frames) ## (B, 1, T)
            n_full = int(bs * self.full_codebook_rate)
            n_dropout = int(bs * self.quantizer_dropout)
            n_imps = int(bs) - n_full - n_dropout
            
            dropout_mask = generate_mask_hard(dropout[:n_dropout], self.n_codebooks) ## (B, Nq, T)
            mask_imp[n_imps:n_imps+n_dropout] = dropout_mask.detach()
            mask_imp[n_imps+n_dropout:] = 1.0
        else:
            n_imps = bs
        
        ### Apply mask
        # mask_imp: (B, Nq, T)
        z_q_is_stack = torch.stack(z_q_is, dim=1) # (B, Nq, D, T)
        z_q = torch.sum(z_q_is_stack * mask_imp[:, :, None, :], dim=1, keepdim=False) # (B, D, T)
        commitment_loss = (commitment_loss * mask_imp.detach()).sum(dim=1).mean() # (B, Nq, T)
        codebook_loss = (codebook_loss * mask_imp.detach()).sum(dim=1).mean()
        
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        if imp_map is not None:
            imp_map_out = imp_map[:n_imps]
        else:
            imp_map_out = None
            
        out_dict = {
            "z_q": z_q,
            "z_q_is": z_q_is_stack,
            "codes": codes,
            "latents": latents,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "imp_map": imp_map_out,
            "mask_imp": mask_imp,
        }
        
        return out_dict
    
    def from_codes(self, codes: torch.Tensor, return_z_q_is=False):
        raise NotImplementedError
    
    def from_latents(self, latents: torch.Tensor):
        raise NotImplementedError

class DAC_MOE(BaseModel, CodecMixin):
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
        level_min: float=1, ## minimum Scale factor
        level_max: float=1, ## maximum Scale factor
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
            self.quantizer = MOEResidualVectorQuantize(
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
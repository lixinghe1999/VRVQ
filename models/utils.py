import torch
import math
from einops import rearrange
import numpy as np
# import torchmetrics
import torchmetrics
import audiotools
from copy import deepcopy
import time 

def logcosh(alpha, pmk):
    """
    For stable training, 
    we divide the calculation into two cases: pmk >= 0 and pmk < 0.
    """
    EPS = 1e-10

    mask1 = pmk >= 0
    pmk1 = pmk * mask1.detach()
    numer1 = math.exp(alpha) + torch.exp(-2*pmk1*alpha)
    denom1 = torch.exp(alpha*(-2*pmk1+1)) + 1
    mask_smooth1 = (torch.log(numer1 + EPS) - torch.log(denom1 + EPS)) / (2*alpha) + 0.5
    

    mask2 = pmk < 0
    pmk2 = pmk * mask2.detach()
    numer2 = torch.exp(alpha*(2*pmk2+1)) + 1
    denom2 = math.exp(alpha) + torch.exp(alpha*2*pmk2)
    mask_smooth2 = (torch.log(numer2 + EPS) - torch.log(denom2 + EPS)) / (2*alpha) + 0.5
    
    mask_smooth = mask_smooth1 * mask1 + mask_smooth2 * mask2
    return mask_smooth


def generate_mask_ste(x, nq, alpha=1):
    device = x.device
    nqs = torch.arange(nq, dtype=torch.float).to(device) # (nq, ), [0, 1, ..., nq-1]
    nqs = rearrange(nqs, 'n -> 1 n 1')
    xmnq = x - nqs # (B, nq, T)
    mask_smooth = logcosh(alpha, xmnq)
    mask_quant = torch.where(xmnq>=0, torch.ones_like(xmnq), torch.zeros_like(xmnq)).float()
    final_mask = mask_smooth + (mask_quant - mask_smooth).detach()
    return final_mask

def generate_mask_hard(x, nq):
    device = x.device
    nqs = torch.arange(nq, dtype=torch.float).to(device) # (nq, ), [0, 1, ..., nq-1]
    nqs = rearrange(nqs, 'n -> 1 n 1')
    xmnq = x - nqs # (B, nq, T)
    mask_quant = torch.where(xmnq>=0, torch.ones_like(xmnq), torch.zeros_like(xmnq)).float()
    return mask_quant


def cal_bpf_from_mask(mask, bits_per_codebook):
    """
    mask: (B, Nq, Frames)
    bits_per_codebook: (Nq, )
    """
    bits_per_codebook = torch.tensor(bits_per_codebook, device=mask.device) ## (Nq, )
    bits_per_codebook = rearrange(bits_per_codebook, 'nq -> 1 nq 1')
    mask_bits = mask * bits_per_codebook
    bpf = torch.sum(mask_bits) / (mask.shape[0] * mask.shape[2])
    return bpf.item()


def cal_entropy(bincount_list):
    n_codebooks = len(bincount_list)
    entropy_list = []
    pct_list = []
    for i in range(n_codebooks):
        bit = math.ceil(math.log2(bincount_list[i].shape[0]))
        counts = bincount_list[i]
        counts = (counts / counts.sum()).clamp(1e-10) ## 각 index의 확률
        entropy_i = -(counts * counts.log()).sum().item() * np.log2(np.e) 
        pct_i = entropy_i / bit
        entropy_list.append(entropy_i)
        pct_list.append(pct_i)
    return entropy_list, pct_list


def cal_metrics(recons, signal, state=None, loss_fn="mel"):
    if loss_fn == "mel":
        return state.mel_loss(recons, signal).item()
    elif loss_fn == "stft":
        return state.stft_loss(recons, signal).item()
    elif loss_fn == "waveform":
        return state.waveform_loss(recons, signal).item()
    elif loss_fn == "SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        if recons.abs().max() == 0 or signal.abs().max() == 0:
            return np.nan  
        result = torchmetrics.functional.signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SI-SDR":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "L1":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.mean_absolute_error(recons, signal)
        result = result.item()
        return result
    elif loss_fn == "SI-SNR":
        recons = recons.audio_data
        signal = signal.audio_data 
        result = torchmetrics.functional.scale_invariant_signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "SNR":
        recons = recons.audio_data
        signal = signal.audio_data
        result = torchmetrics.functional.signal_noise_ratio(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "ViSQOL":
        ## resample to 48k
        result = audiotools.metrics.quality.visqol(recons, signal)
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result
    elif loss_fn == "ViSQOL-speech":
        ## resample to 16k
        result = audiotools.metrics.quality.visqol(recons, signal, "speech")
        if isinstance(result, torch.Tensor):
            result = result.item()
        return result
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std

import torch
import math
from einops import rearrange
import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio import ShortTimeObjectiveIntelligibility as STOI
import torchmetrics
import audiotools
from torchmetrics.audio import DeepNoiseSuppressionMeanOpinionScore as DNSMOS
from torchmetrics.audio import NonIntrusiveSpeechQualityAssessment as NISQA
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


def cal_metrics(recons, signal, state, loss_fn="mel"):
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
    elif loss_fn == "DAC-SISDR":
        return state.dac_sisdr_loss(signal, recons).item()
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
    elif loss_fn == "PESQ":
        sr = signal.sample_rate
        if sr != 16000:
            signal = signal.clone().resample(16000)
            recons = recons.clone().resample(16000)
        recons = recons.audio_data
        signal = signal.audio_data
        pesq = PESQ(16000, 'wb')
        result = pesq(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "STOI":
        sr = signal.sample_rate
        if sr != 16000:
            signal = signal.clone().resample(16000)
            recons = recons.clone().resample(16000)
        recons = recons.audio_data
        signal = signal.audio_data
        stoi = STOI(16000, extended=False)
        result = stoi(recons, signal)
        result = result.mean().item()
        return result
    elif loss_fn == "ESTOI":
        sr = signal.sample_rate
        if sr != 16000:
            signal = signal.clone().resample(16000)
            recons = recons.clone().resample(16000)
        recons = recons.audio_data
        signal = signal.audio_data
        stoi = STOI(16000, extended=True)
        result = stoi(recons, signal)
        result = result.mean().item()
        return result
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")


def mean_std(data):
    data = data[~np.isnan(data)]
    mean = np.mean(data)
    std = np.std(data)
    return mean, std


def cal_metrics_fullxxxx(recons, signal, noisy, state, cal_visqol=True):
    """
    compute the metrics. at once
    recons_t: (1, 1, T)
    """
    # print("#### Signal Sample rate: ", signal.sample_rate)
    # print("#### Recons Sample rate: ", recons.sample_rate)
    st = time.time()
    recons_t = recons.audio_data
    signal_t = signal.audio_data
    sr_ori = signal.sample_rate
    assert recons_t.shape[0] == 1 and recons_t.shape[1]==1
    
    mel_loss = state.mel_loss(recons, signal).item()
    stft_loss = state.stft_loss(recons, signal).item()
    wavefrom_loss = state.waveform_loss(recons, signal).item()

    if recons_t.abs().max() == 0 or signal_t.abs().max() == 0:
        sdr_tm = np.nan
    else:
        sdr_tm = torchmetrics.functional.signal_distortion_ratio(recons_t, signal_t).mean().item()
    
    si_sdr_tm = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons_t, signal_t).mean().item()
    si_snr_tm = torchmetrics.functional.scale_invariant_signal_noise_ratio(recons_t, signal_t).mean().item()
    snr_tm = torchmetrics.functional.signal_noise_ratio(recons_t, signal_t).mean().item()


    if signal.sample_rate != 16000:
        signal_16k = deepcopy(signal).resample(16000)
        recons_16k = deepcopy(recons).resample(16000)
    else:
        signal_16k = signal
        recons_16k = recons
    pesq_tm = PESQ(16000, 'wb')(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    stoi_tm = STOI(16000, extended=False)(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    estoi_tm = STOI(16000, extended=True)(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    # import pdb; pdb.set_trace()

    if cal_visqol:
        visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    ### From here, adapted from sgmse
    if noisy is not None:
        x_hat = recons.audio_data.detach().cpu().numpy().squeeze()
        x = signal.audio_data.detach().cpu().numpy().squeeze()
        n = noisy.audio_data.detach().cpu().numpy().squeeze()

        si_sdr_sg, si_sir_sg, si_sar_sg = energy_ratios(x_hat, x, n)
        si_sdr_sg, si_sir_sg, si_sar_sg = si_sdr_sg.item(), si_sir_sg.item(), si_sar_sg.item()
    else:
        si_sdr_sg, si_sir_sg, si_sar_sg = None, None, None

    return_dict = {
        "mel_loss": mel_loss,
        "stft_loss": stft_loss,
        "waveform_loss": wavefrom_loss,
        "SDR": sdr_tm,
        "SI-SDR": si_sdr_tm,
        "SI-SNR": si_snr_tm,
        "SNR": snr_tm,
        "PESQ": pesq_tm,
        "STOI": stoi_tm,
        "ESTOI": estoi_tm
    }
    # print("Metrics time: ", time.time()-st)

    ### Non-intrusive
    # st = time.time()
    # dnsmos = DNSMOS(sr_ori, personalized=False)
    # p808_mos, mos_sig, mos_bak, mos_ovr = dnsmos(recons_t.squeeze())

    # dns_dict = {
    #     "p808_MOS":p808_mos.item(),
    #     "MOS_SIG":mos_sig.item(),
    #     "MOS_BAK":mos_bak.item(),
    #     "MOS_OVR":mos_ovr.item()
    # }
    # return_dict.update(dns_dict)
    # print("DNSMOS time: ", time.time()-st)

    ### NISQA
    st = time.time()
    nisqa = NISQA(sr_ori)
    nisqa_overall_mos, nisqa_noisiness, nisqa_discontinuity, nisqa_coloration, nisqa_loudness = \
        nisqa(recons_t.squeeze())
        
    nisqa_dict = {
        "NISQA_overall_MOS":nisqa_overall_mos.item(),
        "NISQA_noisiness":nisqa_noisiness.item(),
        "NISQA_discontinuity":nisqa_discontinuity.item(),
        "NISQA_coloration":nisqa_coloration.item(),
        "NISQA_loudness":nisqa_loudness.item()
    }
    return_dict.update(nisqa_dict)
    # print("NISQA time: ", time.time()-st)

    if noisy is not None:
        return_dict["SI-SDR-sg"] = si_sdr_sg
        return_dict["SI-SIR-sg"] = si_sir_sg
        return_dict["SI-SAR-sg"] = si_sar_sg

    if cal_visqol:
        return_dict["ViSQOL-speech"] = visqol_speech
    
    return return_dict

    


def cal_metrics_visqol(recons, signal):
    """
    compute the metrics. at once
    recons_t: (1, 1, T)
    """

    if signal.sample_rate != 16000:
        # assert False
        # signal_16k = signal.resample(16000)
        # recons_16k = recons.resample(16000)
        signal_16k = deepcopy(signal).resample(16000)
        recons_16k = deepcopy(recons).resample(16000)
    else:
        signal_16k = signal
        recons_16k = recons
        
    visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    return_dict = {}
    return_dict["ViSQOL-speech"] = visqol_speech
    
    return return_dict

    


###################

def cal_metrics_full(recons, signal, noisy, state, cal_visqol=True):
    """
    compute the metrics. at once
    recons_t: (1, 1, T)
    """
    # print("#### Signal Sample rate: ", signal.sample_rate)
    # print("#### Recons Sample rate: ", recons.sample_rate)
    st = time.time()
    recons_t = recons.audio_data
    signal_t = signal.audio_data
    sr_ori = signal.sample_rate
    assert recons_t.shape[0] == 1 and recons_t.shape[1]==1
    
    # mel_loss = state.mel_loss(recons, signal).item()
    # stft_loss = state.stft_loss(recons, signal).item()
    # wavefrom_loss = state.waveform_loss(recons, signal).item()

    # if recons_t.abs().max() == 0 or signal_t.abs().max() == 0:
    #     sdr_tm = np.nan
    # else:
    #     sdr_tm = torchmetrics.functional.signal_distortion_ratio(recons_t, signal_t).mean().item()
    
    # si_snr_tm = torchmetrics.functional.scale_invariant_signal_noise_ratio(recons_t, signal_t).mean().item()
    # snr_tm = torchmetrics.functional.signal_noise_ratio(recons_t, signal_t).mean().item()
    si_sdr_tm = torchmetrics.functional.scale_invariant_signal_distortion_ratio(recons_t, signal_t).mean().item()


    if signal.sample_rate != 16000:
        signal_16k = deepcopy(signal).resample(16000)
        recons_16k = deepcopy(recons).resample(16000)
    else:
        signal_16k = signal
        recons_16k = recons
    pesq_tm = PESQ(16000, 'wb')(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    stoi_tm = STOI(16000, extended=False)(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    estoi_tm = STOI(16000, extended=True)(recons_16k.audio_data, signal_16k.audio_data).mean().item()
    # import pdb; pdb.set_trace()

    # if cal_visqol:
    #     visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    
    
    
    ### From here, adapted from sgmse
    # if noisy is not None:
    #     x_hat = recons.audio_data.detach().cpu().numpy().squeeze()
    #     x = signal.audio_data.detach().cpu().numpy().squeeze()
    #     n = noisy.audio_data.detach().cpu().numpy().squeeze()

    #     si_sdr_sg, si_sir_sg, si_sar_sg = energy_ratios(x_hat, x, n)
    #     si_sdr_sg, si_sir_sg, si_sar_sg = si_sdr_sg.item(), si_sir_sg.item(), si_sar_sg.item()
    # else:
    #     si_sdr_sg, si_sir_sg, si_sar_sg = None, None, None

    return_dict = {
        # "mel_loss": mel_loss,
        # "stft_loss": stft_loss,
        # "waveform_loss": wavefrom_loss,
        # "SDR": sdr_tm,
        "SI-SDR": si_sdr_tm,
        # "SI-SNR": si_snr_tm,
        # "SNR": snr_tm,
        "PESQ": pesq_tm,
        "STOI": stoi_tm,
        "ESTOI": estoi_tm
    }
    # print("Metrics time: ", time.time()-st)

    ### Non-intrusive
    # st = time.time()
    # dnsmos = DNSMOS(sr_ori, personalized=False)
    # p808_mos, mos_sig, mos_bak, mos_ovr = dnsmos(recons_t.squeeze())

    # dns_dict = {
    #     "p808_MOS":p808_mos.item(),
    #     "MOS_SIG":mos_sig.item(),
    #     "MOS_BAK":mos_bak.item(),
    #     "MOS_OVR":mos_ovr.item()
    # }
    # return_dict.update(dns_dict)
    # print("DNSMOS time: ", time.time()-st)

    ### NISQA
    st = time.time()
    nisqa = NISQA(sr_ori)
    nisqa_overall_mos, nisqa_noisiness, nisqa_discontinuity, nisqa_coloration, nisqa_loudness = \
        nisqa(recons_t.squeeze())
        
    nisqa_dict = {
        "NISQA_overall_MOS":nisqa_overall_mos.item(),
        "NISQA_noisiness":nisqa_noisiness.item(),
        "NISQA_discontinuity":nisqa_discontinuity.item(),
        "NISQA_coloration":nisqa_coloration.item(),
        "NISQA_loudness":nisqa_loudness.item()
    }
    return_dict.update(nisqa_dict)
    # print("NISQA time: ", time.time()-st)
    
    ### VISQOL?
    # visqol_speech = audiotools.metrics.quality.visqol(recons_16k, signal_16k, "speech").mean().item()
    # return_dict["ViSQOL-speech"] = visqol_speech

    # if noisy is not None:
    #     return_dict["SI-SDR-sg"] = si_sdr_sg
    #     return_dict["SI-SIR-sg"] = si_sir_sg
    #     return_dict["SI-SAR-sg"] = si_sar_sg

    # if cal_visqol:
    #     return_dict["ViSQOL-speech"] = visqol_speech
    
    return return_dict
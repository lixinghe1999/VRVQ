import typing
from typing import List, Union

import torch
import torch.nn.functional as F
import torchaudio as ta
from torchaudio import transforms as T
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
from einops import rearrange


class L1Loss(nn.L1Loss):
    """L1 Loss between AudioSignals. Defaults
    to comparing ``audio_data``, but any
    attribute of an AudioSignal can be used.

    Parameters
    ----------
    attribute : str, optional
        Attribute of signal to compare, defaults to ``audio_data``.
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)

    def forward(self, x: AudioSignal, y: AudioSignal):
        """
        Parameters
        ----------
        x : AudioSignal
            Estimate AudioSignal
        y : AudioSignal
            Reference AudioSignal

        Returns
        -------
        torch.Tensor
            L1 loss between AudioSignal attributes.
        """
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)
    
    
class L2Loss(nn.MSELoss):
    def __init__(self, attribute: str = "audio_data", weight: float = 1.0, **kwargs):
        self.attribute = attribute
        self.weight = weight
        super().__init__(**kwargs)
    
    def forward(self, x: AudioSignal, y: AudioSignal):
        if isinstance(x, AudioSignal):
            x = getattr(x, self.attribute)
            y = getattr(y, self.attribute)
        return super().forward(x, y)



class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/distance.py
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1) # (nb, nt, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1) # (nb, nt, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError("Invalid reduction type.")
        return sdr


class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for s in self.stft_params:
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss


class MelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        # loss_fn: Union[typing.Callable, List[typing.Callable]] = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        # pow: Union[float, List[float]] = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
        reduction='mean',
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
            
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.reduction=reduction
        self.pow = pow
        
        # import pdb; pdb.set_trace()
        # print('pow: ', pow) ## Argbind로 어떻게 설정?
        
        # if not isinstance(pow, list):
        #     self.pow = [pow]
        # else:
        #     self.pow = pow
    
    def forward(self, x: AudioSignal, y: AudioSignal, levels=None):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        # import pdb; pdb.set_trace()
        
        if levels is None:
            self.loss_fn.reduction='mean'
        else:
            self.loss_fn.reduction='none'
            levels = rearrange(levels, 'b 1 1 -> b')
        

        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            if levels is None:
                loss += self.log_weight * self.loss_fn(
                    x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                    y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                )
                loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
            else:
                ## levels: (B, 1, 1), >=0.5 
                loss_temp = 0
                level_weight = 1/levels
                
                losses_log = self.loss_fn(
                    x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                    y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                )  ## => (B, 1, mel, frames)
                ## Mean by dimensions except batch
                losses_log = losses_log.mean(dim=[-1, -2, -3])
                # losses_log = losses_log * level_weight
                # loss += losses_log.mean()
                loss_temp += losses_log
                
                losses = self.loss_fn(x_mels, y_mels)
                losses = losses.mean(dim=[-1, -2, -3])
                loss_temp += self.mag_weight * losses
                
                loss += (level_weight * loss_temp).mean()
                
                
        #     print("Loss in final : ", loss.mean())
        # print("#### Melspec final loss ####")
        # print(loss)
        return loss


class GANLoss(nn.Module):
    """
    Computes a discriminator loss, given a discriminator on
    generated waveforms/spectrograms compared to ground truth
    waveforms/spectrograms. Computes the loss for both the
    discriminator and the generator in separate functions.
    """

    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def forward(self, fake, real):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        # import pdb; pdb.set_trace()
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        return loss_d

    def generator_loss(self, fake, real):
        ## fake, real: (B, 1, T)
        # import pdb; pdb.set_trace()
        d_fake, d_real = self.forward(fake, real)
        ## d_fake, d_real: features. 2d list.
        ## length = 8개의 discriminator.
        ## 

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())
        return loss_g, loss_feature


"""
Framewise-loss
"""

class SISDRLossFramewise(nn.Module):
    def __init__(
        self,
        scaling: int = True,
        reduction: str = "none",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()
    
        self.sisdr_loss = SISDRLoss(scaling, reduction, zero_mean, clip_min, weight)
        
    def forward(self, x, y, window_size: int = 512):
        eps = 1e-8
        # nb, nc, nt
        # nb, nc, nt = x.audio_data.size()
        nb, nc, nt = x.size()
        assert nt % window_size == 0, f"nt: {nt}, window_size: {window_size}"
        n_frames = nt // window_size
        # loss_output = torch.zeros(nb, n_frames)
        
        x_framewise = rearrange(x, 'b c (f w) -> (b f) c w', w=window_size)
        y_framewise = rearrange(y, 'b c (f w) -> (b f) c w', w=window_size)
        
        loss_output = self.sisdr_loss(x_framewise, y_framewise)
        # import pdb; pdb.set_trace()
        loss_output = rearrange(loss_output, '(b f) 1 -> b f', f=n_frames)
        return loss_output
        

class L1LossFramewise(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)
        
    def forward(self, x, y, window_size: int = 512):
        # nb, nc, nt
        # nb, nc, nt = x.audio_data.size()
        nb, nc, nt = x.size()
        assert nt % window_size == 0, f"nt: {nt}, window_size: {window_size}"
        n_frames = nt // window_size
        # loss_output = torch.zeros(nb, n_frames)
        
        x_framewise = rearrange(x, 'b c (f w) -> (b f) c w', w=window_size)
        y_framewise = rearrange(y, 'b c (f w) -> (b f) c w', w=window_size)
        loss_output = self.l1_loss(x_framewise, y_framewise)
        loss_output = loss_output.mean(dim=-1) # (BF, 1)
        # import pdb; pdb.set_trace()
        loss_output = rearrange(loss_output, '(b f) 1 -> b f', f=n_frames)
        return loss_output
    

class MelSpectrogramLossFramewise(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [150, 80],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    ### init with base.yml except n_mels, windws lengths
    def __init__(
        self,
        n_mels: List[int] = [160, 80, 40, 20],
        window_lengths: List[int] = [512, 512, 512, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0, # 1.0->0.0
        log_weight: float = 1.0,
        pow: float = 1.0, # 2.0->1.0
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0,0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
        reduction='none',
        sr=44100,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                # hop_length=w // 4,
                # hop_length=w,
                hop_length=512,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
        # self.reduction=reduction
        self.sr = sr
        self.loss_fn.reduction=reduction
        
        self.meltransform_list = nn.ModuleList([
            T.MelSpectrogram(sample_rate=sr,
                                    n_mels=n_mel,
                                    n_fft=w,
                                    hop_length=w,
                                    f_min=fmin,
                                    f_max=fmax,
                                    center=False)
            for n_mel, w, fmin, fmax in zip(n_mels, window_lengths, mel_fmin, mel_fmax)
        ])
        # self.meltransform_list = [
        #     T.MelSpectrogram(sample_rate=sr,
        #                             n_mels=n_mel,
        #                             n_fft=w,
        #                             hop_length=w,
        #                             f_min=fmin,
        #                             f_max=fmax,
        #                             center=False).to('cuda')
        #     for n_mel, w, fmin, fmax in zip(n_mels, window_lengths, mel_fmin, mel_fmax)
        # ]

    def forward(self, x, y, window_size=None):
        loss = 0.0

        for meltransform in self.meltransform_list:
            loss_in_loop = 0.0
            x_mels = meltransform(x)
            y_mels = meltransform(y)
            loss_in_loop += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            if self.mag_weight > 0:
                loss_in_loop += self.mag_weight * self.loss_fn(x_mels, y_mels)    
            # print("Loss in loop mean: ", loss_in_loop.mean())
            loss_in_loop = loss_in_loop.mean(dim=-2) # (B, 1, mel, frames) -> (B, 1, frames)
            loss += loss_in_loop
        
        loss = loss.squeeze(1)
        # print("#### Melspec Framewise loss ####")
        # print(loss)
        return loss
    

## Duplicate Melspectrogramloss for argbind
class MelSpectrogramLossDuplicate(nn.Module):

    def __init__(
        self,
        n_mels: List[int] = [150, 80],
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.MSELoss(),
        # loss_fn: Union[typing.Callable, List[typing.Callable]] = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        # pow: Union[float, List[float]] = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0.0, 0.0],
        mel_fmax: List[float] = [None, None],
        window_type: str = None,
        reduction='mean',
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
            
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.reduction=reduction
        self.pow = pow
        
        # import pdb; pdb.set_trace()
        # print('pow: ', pow) ## Argbind로 어떻게 설정?
        
        # if not isinstance(pow, list):
        #     self.pow = [pow]
        # else:
        #     self.pow = pow
    
    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        loss = 0.0
        # import pdb; pdb.set_trace()
        

        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "window_type": s.window_type,
            }
            x_mels = x.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)
            y_mels = y.mel_spectrogram(n_mels, mel_fmin=fmin, mel_fmax=fmax, **kwargs)

            loss += self.log_weight * self.loss_fn(
                x_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
                y_mels.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x_mels, y_mels)
        #     print("Loss in final : ", loss.mean())
        # print("#### Melspec final loss ####")
        # print(loss)
        return loss
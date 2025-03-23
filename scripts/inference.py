import os; opj=os.path.join
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
from PIL import Image

from audiotools import AudioSignal
from data.loaders import AudioLoader


from models.dac_vrvq import DAC_VRVQ
from models.utils import cal_bpf_from_mask, generate_mask_hard, cal_metrics
import argbind

import matplotlib.pyplot as plt
import librosa.display
import math
import json


DAC_VRVQ = argbind.bind(DAC_VRVQ)

@argbind.bind(without_prefix=True)
def inference(
    args,
    ckpt_path:str=None,
    ckpt_dir:str=None,
    tag:str=None,
    save_result_dir:str=None,
    data_dir:str=None,
    num_examples: int = 30,
    device:str="cpu",
    ):
    config_path = args["args.load"]
    exp_name = config_path.replace('conf/', '').replace('.yml', '')

    if ckpt_path is None:
        ckpt_path = opj(ckpt_dir, exp_name, tag, 'dac_vrvq', 'weights.pth')

    ## Load model
    model = DAC_VRVQ()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    model.to(device)
    
    ## Audio loader
    audio_loader = AudioLoader(
        sources=[data_dir],
        shuffle=False,
    )
    for idx in range(num_examples):
        # Load random audio
        state = np.random.RandomState(idx)
        item = audio_loader(
            state=state,
            sample_rate=model.sample_rate,
            duration=10,
            num_channels=1,
        )
        signal = item['signal']
        signal = signal.to(device)
        path = item['path']    
                
        level_list = [0.2, 0.35, 0.5, 0.6, 0.8, 1, 2, 4] 
        level_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.5, 3]
        save_results(model, signal.audio_data, level_list, save_result_dir)
        print("Saved results for ", idx)

def save_results(model, input_tensor, level_list, save_result_dir):
    """
    plot results
    """
    metadata = {}
    os.makedirs(save_result_dir, exist_ok=True)
    save_idx = 0
    while True:
        save_dir = opj(save_result_dir, f"{save_idx}")
        if os.path.exists(save_dir):
            save_idx += 1
        else:
            os.makedirs(save_dir)
            break

    with torch.no_grad():
        n_q = model.n_codebooks
        input_tensor = model.preprocess(input_tensor, model.sample_rate)
        enc_out = model.encode(input_tensor, n_quantizers=None, level=1) ## level value is dummy value
        imp_map = enc_out['imp_map']
        
    ## Save imp_map
    for level in level_list:
        level_scaled = level*n_q
        imp_map_scaled = imp_map * level_scaled
        mask_imp = generate_mask_hard(imp_map_scaled, nq=n_q) # (B, Nq, T)
        z_q_is = enc_out["z_q_is"] # (B, Nq, D, T)
        z_q = torch.sum(z_q_is * mask_imp[:,:,None,:], dim=1, keepdim=False) # (B, D, T)
        with torch.no_grad():
            recon = model.decode(z_q)
        save_mask_imp(mask_imp, level_scaled, save_dir)
        
        
        ## Save reconstructed audio
        recon_signal = AudioSignal(recon, sample_rate=model.sample_rate)
        input_signal = AudioSignal(input_tensor, sample_rate=model.sample_rate)
        sisdr = cal_metrics(recon_signal, input_signal, loss_fn="SI-SDR")
        bpf = cal_bpf_from_mask(mask_imp, 
                                bits_per_codebook=[10]*n_q) ## each codebook has 1024 indices
        kbps = bpf * math.floor(model.sample_rate / model.hop_length) / 1000
        filename = f"recon_{level_scaled:.2f}.wav"
        recon_signal.to('cpu').write(opj(save_dir, filename))
        
        metadata[f"level_{level_scaled:.2f}"] = {
            "sisdr": sisdr,
            "kbps": kbps,
        }
    
    with open(opj(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    ## Input Save Spectrograms
    fig, ax = plt.subplots(figsize=(9, 5))    
    signal_audio = input_signal[0].cpu()
    ref_re = signal_audio.magnitude.max()
    logmag_re = signal_audio.log_magnitude(ref_value=ref_re)
    logmag_re = logmag_re.numpy()[0][0]
    librosa.display.specshow(
        logmag_re,
        sr=model.sample_rate,
        y_axis='linear',
        x_axis='time',
        ax=ax,
    )
    # ax.set_title("Input")
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Freq. (Hz)", fontsize=14)
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    plt.savefig(opj(save_dir, "input.png"))
    plt.close()
    input_signal.to('cpu').write(opj(save_dir, "input.wav"))


def save_mask_imp(mask_imp, level, save_dir):
    """
    save imp_map_scaled as images
    imp_map_scaled: (1, Nq, T), binary mask
    """
    nq = mask_imp.shape[1]
    mask_imp = mask_imp.squeeze(0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(mask_imp, cmap="viridis", aspect='auto', interpolation="none")
    ax.set_yticks(np.arange(0, nq))
    ax.set_yticklabels(np.arange(1, nq+1), fontsize=20)
    ax.invert_yaxis()
    
    ax.set_xticks([])
    ax.set_xticklabels([])

    plt.tight_layout()
    plt.savefig(opj(save_dir, f"imp_map_{level:.2f}.png"))
    plt.close()
    


if __name__=="__main__":
    
    args = argbind.parse_args()
    with argbind.scope(args):
        inference(args)
    
    

""" ## Single Audio Inference Example

if isinstance(audio_file, str):
    audio = AudioSignal(audio_file)
else:
    audio = audio_file
# audio = AudioSignal(audio_file)
audio = audio.to_mono()
audio = audio.resample(model.sample_rate)
assert audio.sample_rate == model.sample_rate, f"Sample rate mismatch: {audio.sample_rate} vs {model.sample_rate}"
audio = audio.to(device)
audio_tensor = audio.audio_data # (1, 1, T)


# Encode
# level: range in model.level_min, model.level_max
# n_quantizers: if specified, the number of quantizers to use. If None, use all quantizers. i.e., it becomes CBR.
# encoded: "z_q", "codes", "latents", "commitment_loss", "codebook_loss", "imp_map", "mask_imp"

with torch.no_grad():
    level = 1 # Dummy value
    audio_tensor = model.preprocess(audio_tensor, model.sample_rate)
    encoded = model.encode(audio_tensor, n_quantizers=None, level=level)
    # decoded = model.decode(encoded['z_q'])
    codes = encoded['codes'] # (B, Nq, T)
    z_q_is = encoded['z_q_is'] # (B, Nq, D, T)
    imp_map = encoded['imp_map'] # (B, Nq, T)
    # decoded = model.decode(encoded['z_q'])

## Results
print("Audio: ", audio_file)
print("Audio Shape: ", audio_tensor.shape)
print("z_q: ", encoded['z_q'].shape)
print("codes: ", encoded['codes'].shape)
print("imp_map: ", encoded['imp_map'].shape)
print("reconstructed: ", decoded.shape)

recon_signal = AudioSignal(decoded, sample_rate=model.sample_rate)
input_signal = AudioSignal(audio_tensor, sample_rate=model.sample_rate)
si_sdr = cal_metrics(recon_signal, input_signal, loss_fn="SI-SDR")
"""
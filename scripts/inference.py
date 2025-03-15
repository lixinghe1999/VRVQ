import os; opj=os.path.join
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Union

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


DAC_VRVQ = argbind.bind(DAC_VRVQ)

@argbind.bind(without_prefix=True)
def inference(
    args,
    audio_file:Union[str, AudioSignal],
    ckpt_path:str=None,
    # ckpt_dir:str=None, ## /data2/yoongi/vrvq_github
    ckpt_dir:str=None,
    tag:str=None,
    save_result_dir:str=None,
    device:str="cpu",
    # device:Union[str, int]=None,
    ):
    config_path = args["args.load"]
    exp_name = config_path.replace('conf/', '').replace('.yml', '')
    # print(f"tag: {tag} || save_result_dir: {save_result_dir} || device: {device}")
    # assert False
    # import pdb; pdb.set_trace()
    # if device != 'cpu':
    #     if device != 'cuda':
    #         device = f'cuda:{device}'
    #     else:
    #         device = 'cuda:0'

    # device = torch.device(device)
    # import pdb; pdb.set_trace()
    # assert device is not None
    if ckpt_path is None:
        ckpt_path = opj(ckpt_dir, exp_name, tag, 'dac_vrvq', 'weights.pth')
    # import pdb; pdb.set_trace()

    ## Load model
    model = DAC_VRVQ()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    model.to(device)

    ## Load audio
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

    """
    Encode
    level: range in model.level_min, model.level_max
    n_quantizers: if specified, the number of quantizers to use. If None, use all quantizers. i.e., it becomes CBR.

    encoded: "z_q", "codes", "latents", "commitment_loss", "codebook_loss", "imp_map", "mask_imp"
    """
    with torch.no_grad():
        level = 1 # Dummy value
        audio_tensor = model.preprocess(audio_tensor, model.sample_rate)
        encoded = model.encode(audio_tensor, n_quantizers=None, level=level)
        # decoded = model.decode(encoded['z_q'])
        codes = encoded['codes'] # (B, Nq, T)
        z_q_is = encoded['z_q_is'] # (B, Nq, D, T)
        imp_map = encoded['imp_map'] # (B, Nq, T)
        
    
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
    level_list = [0.2, 0.35, 0.5,0.6, 0.8, 1, 2, 4]
    
    save_results(model, audio_tensor, level_list, save_result_dir)


def save_results(model, input_tensor, level_list, save_result_dir):
    """
    plot results
    """
    os.makedirs(save_result_dir, exist_ok=True)
    save_idx = 0
    while True:
        save_dir = opj(save_result_dir, f"{save_idx}")
        if os.path.exists(save_dir):
            save_idx += 1
        else:
            os.makedirs(save_dir)
            break
        
    ## Save Spectrograms
    fig, ax = plt.subplots(figsize=(10, 5))
    signal = AudioSignal(input_tensor, sample_rate=model.sample_rate)
    signal_audio = signal[0].cpu()
    ref_re = signal_audio.magnitude.max()
    logmag_re = signal_audio.log_magnitude(ref_value=ref_re)
    logmag_re = logmag_re.numpy()[0][0]
    librosa.display.specshow(
        logmag_re,
        sr=model.sample_rate,
        x_axis='time',
        ax=ax,
    )
    ax.set_title("Input")
    plt.savefig(opj(save_dir, "input.png"))
    plt.close()

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
        recon = model.decode(z_q)
        save_mask_imp(mask_imp, level_scaled, save_dir)
    import pdb; pdb.set_trace()
        

def save_mask_imp(mask_imp, level, save_dir):
    """
    save imp_map_scaled as images
    imp_map_scaled: (1, Nq, T), binary mask
    """
    nq = mask_imp.shape[1]
    mask_imp = mask_imp.squeeze(0).detach().cpu().numpy()
    # mask_imp = np.flipud(mask_imp)
    # fig, ax = plt.figure(figsize=(10, 5))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(mask_imp, cmap="viridis", aspect='auto', interpolation="none")
    ax.set_yticks(np.arange(0, nq))
    ax.invert_yaxis()

    plt.savefig(opj(save_dir, f"imp_map_{level:.2f}.png"))
    plt.close()
    # import pdb; pdb.set_trace()


# def plot_imp_map(imp_map_scaled, level, save_dir):
#     """
#     Save imp_map_scaled as a binary image without interpolation using PIL.
#     imp_map_scaled: (1, Nq, T), binary mask tensor.
#     """
#     # squeeze and vertical flip
#     imp_map_scaled = imp_map_scaled.squeeze(0).detach().cpu().numpy()
#     imp_map_scaled = np.flipud(imp_map_scaled)
    
#     # Convert binary mask (assuming values in [0,1]) to 0 or 255
#     binary_img = (imp_map_scaled * 255).astype(np.uint8)
    
#     # Create the save path
#     save_path = os.path.join(save_dir, f"imp_map_{level:.2f}.png")
    
#     # Save image using PIL in grayscale mode
#     Image.fromarray(binary_img, mode='L').save(save_path)

    
    
    


if __name__=="__main__":
    
    data_dir = '/data2/yoongi/dataset/daps/test'
    sample_rate = 44100
    duration=3
    state = np.random.RandomState(0)
    audio_loader = AudioLoader(
        sources=[data_dir],
        shuffle=False,
    )
    
    item = audio_loader(
        state=state,
        sample_rate=sample_rate,
        duration=duration,
        num_channels=1,
    )
    signal = item['signal']
    path = item['path']
    
    # import pdb; pdb.set_trace()
    
    args = argbind.parse_args()
    with argbind.scope(args):
        inference(args,
                  audio_file = signal,
                  )
    
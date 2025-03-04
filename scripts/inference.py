import os; opj=os.path.join
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Union

import torch

from audiotools import AudioSignal


from models.dac_vrvq import DAC_VRVQ
from models.utils import cal_bpf_from_mask, generate_mask_hard, cal_metrics
import argbind

# import argparse

# def argument_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--expname', type=str, default='vrvq/vrvq_a2')
#     parser.add_argument('--ckpt-path', type=str, default=None) ## If None, we will load the checkpoint based on the saved dir (i.e., ckpt-dir)
#     parser.add_argument('--ckpt-dir', type=str, default='/data2/yoongi/vrvq_github')
#     parser.add_argument('--tag', type=str, default='latest')
#     parser.add_argument('--device', default='cpu') ## 'cpu' or index of gpu
#     return parser.parse_args()

DAC_VRVQ = argbind.bind(DAC_VRVQ)

@argbind.bind(without_prefix=True)
def inference(
    args,
    audio_file:str=None,
    ckpt_path:str=None,
    # ckpt_dir:str=None, ## /data2/yoongi/vrvq_github
    ckpt_dir:str="/data2/yoongi/vrvq_github",
    tag:str='latest',
    save_result_dir:str=None,
    device:Union[str, int]='cpu',
    ):
    config_path = args["args.load"]
    exp_name = config_path.replace('conf/', '').replace('.yml', '')
    # import pdb; pdb.set_trace()
    if device != 'cpu':
        if device != 'cuda':
            device = f'cuda:{device}'
        else:
            device = 'cuda:0'

    device = torch.device(device)

    if ckpt_path is None:
        ckpt_path = opj(ckpt_dir, exp_name, tag, 'dac_vrvq', 'weights.pth')

    ## Load model
    model = DAC_VRVQ()
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)

    ## Load audio
    audio = AudioSignal(audio_file)
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
    level = 1
    audio_tensor = model.preprocess(audio_tensor, model.sample_rate)
    encoded = model.encode(audio_tensor, n_quantizers=None, level=level)
    decoded = model.decode(encoded['z_q'])
    
    ## Results
    print("Audio: ", audio_file)
    print("Audio Shape: ", audio_tensor.shape)
    print("z_q: ", encoded['z_q'].shape)
    print("codes: ", encoded['codes'].shape)
    print("imp_map: ", encoded['imp_map'].shape)
    print("reconstructed: ", decoded.shape)
    


if __name__=="__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        inference(args)
    
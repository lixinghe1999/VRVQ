# Variable Bitrate Residual Vector Quantization for Audio Coding

This repository contains official implementation of the paper **Variable Bitrate Residual Vector Quantization for Audio Coding**:  


## 📄 Paper Link

- **ICASSP 2025 version**: [IEEE Xplore](https://ieeexplore.ieee.org/document/10889508)
- **NeurIPS 2024 ML Compression Workshop version**: [arXiv](https://arxiv.org/abs/2410.06016)

## 🔊 Audio Samples: 
Importance map and audio samples are available at: [Link](https://sonyresearch.github.io/VRVQ/)

## ⚙️ Environment Setup  
To set up the environment, follow these steps:  

```bash
# Create a conda environment
conda create -n vrvq python=3.9

# Activate the environment
conda activate vrvq

# Install dependencies
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=11.8 -c pytorch -c nvidia  ## We used this command for PyTorch install
pip install -r requirements.txt
```

## Training example
```bash
# ex) bash scripts/script_train.sh $EXP_PATH $GPU

# Single GPU
bash scripts/scripts_train.sh vrvq/vrvq_a2 0

# Multi GPU
bash scripts/scripts_train.sh vrvq/vrvq_a2 0,1
```

## Inference example
Please refer to the `scripts/inference.py` for the inference code.

```bash
# ex) bash scripts/script_inference.sh $EXP_PATH $GPU

bash scripts/scripts_inference.sh vrvq/vrvq_a2 0
```

## Quantization example
```python
import torch
import torchaudio
from models.dac_vrvq import DAC_VRVQ
from models.utils import cal_bpf_from_mask, generate_mask_hard

# Load the audio file
ckpt_path = 'ckpt_dir/exp_name/tag/dac_vrvq/weights.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
model = DAC_VRVQ()
model.load_state_dict(ckpt['model_state_dict'], strict=True)
model.eval()
sample_rate = model.sample_rate # 44100

# Load the audio file
audio_path = 'path/to/audio/file.wav'
wav, sr = torchaudio.load(audio_path) # wav: [1, T], sr: sample rate
wav = wav.squeeze(0)

# Encode and Quantize
level = 1 # Dummy value
nq = model.n_codebooks
wav = model.preprocess(wav, sample_rate)
encoded = model.encode(wav, n_quantizers=None, level=level)

z_q_is = encoded['z_q_is'] # [B, Nq, D, T], quantized latent with full number of quantizers
imp_map = encoded['imp_map'] # importance map
imp_map_scaled = imp_map * level * nq # scaled importance map
mask_imp = generate_mask_hard(imp_map_scaled, nq=n_q) # [B, Nq, T], binary mask
z_q = torch.sum(z_q_is * mask_imp.unsqueeze(2), dim=1) # [B, D, T], quantized latent with

reconstructed = model.decode(z_q)
```


## 📌 Code Base  
Our work is based on **DAC** [1], and our experiments were conducted using its framework.  
Thus, this code is also built upon the DAC repository.  
- **DAC GitHub:** [DAC GitHub Link](https://github.com/descriptinc/descript-audio-codec)  

## 📝 References
[1] Rithesh Kumar, Prem Seetharaman, Alejandro Luebs, Ishaan Kumar, Kundan Kumar,  
**High-Fidelity Audio Compression with Improved RVQGAN**,  
*Advances in Neural Information Processing Systems*, vol. 36, pp. 27980–27993, 2023.  
[Paper Link](https://arxiv.org/abs/2306.06546)

## 📚 Citation

If you find our work useful, please cite:

```bibtex
@INPROCEEDINGS{chae2025vrvq,
  author={Chae, Yunkee and Choi, Woosung and Takida, Yuhta and Koo, Junghyun and Ikemiya, Yukara and Zhong, Zhi and Cheuk, Kin Wai and Martínez-Ramírez, Marco A. and Lee, Kyogu and Liao, Wei-Hsiang and Mitsufuji, Yuki},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Variable Bitrate Residual Vector Quantization for Audio Coding}, 
  year={2025},
  pages={1-5},
  keywords={Training;Adaptation models;Codecs;Audio coding;Vector quantization;Bit rate;Rate-distortion;Estimation;Transforms;Vectors;Neural Audio Codec;Variable Bitrate;Residual Vector Quantization;Rate-Distortion Tradeoff;Importance Map},
  doi={10.1109/ICASSP49660.2025.10889508}}
}

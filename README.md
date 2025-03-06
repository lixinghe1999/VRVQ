# Variable Bitrate Residual Vector Quantization for Audio Coding

This repository contains official implementation of the paper **Variable Bitrate Residual Vector Quantization for Audio Coding**:  
- **Accepted at NeurIPS 2024 Machine Learning and Compression Workshop** 
- **Accepted at ICASSP 2025**  

## 🔊 Audio Samples: 
Importance map and audio samples are available at: Link

## 📌 Code Base  
Our work is based on **DAC** [1], and our experiments were conducted using its framework.  
Thus, this code is also built upon the DAC repository.  
- **DAC GitHub:** [DAC GitHub Link](#)  

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


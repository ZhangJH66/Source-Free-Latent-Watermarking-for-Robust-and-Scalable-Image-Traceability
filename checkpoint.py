"""Unchanged native checkpoint loader. Only load trusted checkpoints."""
import os
import torch
from omegaconf import OmegaConf
from model import WatermarkModel

def load_wm_model(path, device='cuda'):
    if os.path.isdir(path):
        ckpt_dir = path
        ckpt_path = os.path.join(ckpt_dir, "wm_model.ckpt")
    else:
        ckpt_path = path
        ckpt_dir = os.path.dirname(path)
    
    wm_model_config_path = os.path.join(ckpt_dir, "wm_model_config.yaml")
    if not os.path.exists(wm_model_config_path):
        raise FileNotFoundError(f"Config not found: {wm_model_config_path}")
    
    wm_model_config = OmegaConf.load(wm_model_config_path)
    message_length = wm_model_config["wm_enc_config"]["message_length"]
    model = WatermarkModel(**wm_model_config)
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Weights not found: {ckpt_path}")
        
    model_ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(model_ckpt)
    model.eval()
    return model.to(device), message_length

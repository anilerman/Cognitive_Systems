import os
import yaml
import torch
import numpy as np
import random

def load_model_from_checkpoint(checkpoint_path, model_class):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    model = model_class()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_inference_data(data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"No data file found at '{data_path}'")

    data = torch.load(data_path)
    return data

def perform_inference(model, data):
    model.eval()
    with torch.no_grad():
        results = model(data)
    return results

def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(results, output_path)
    print(f"Results saved at {output_path}")


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def save_checkpoint(model, checkpoint_dir, filename):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

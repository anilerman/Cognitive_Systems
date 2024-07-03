# run.py

import os
import torch
from data_loader import create_data_loaders
import model as model_module
import train
import test
from utils import load_config, set_seed, check_gpu
from visualization import plot_metrics, plot_differences
from sklearn.model_selection import KFold
from torchvision import datasets, transforms

def main():
    config_file = 'config.yaml'
    config = load_config(config_file)

    set_seed(config['training']['seed'])
    check_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(config["data"]["classes"])

    k_fold = config['training'].get('k_fold', None)
    if k_fold:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=config['training']['seed'])
        dataset = datasets.ImageFolder(os.path.join(config['data']['data_dir'], 'train'))
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold+1}/{k_fold}")
            train_loader, val_loader, _ = create_data_loaders(config['data'], k_fold=k_fold, fold_idx=fold)
            model_instance = model_module.get_model(config["model"], num_classes).to(device)
            trained_model, metrics = train.train_model(model_instance, train_loader, val_loader, config)
            test.test_model(trained_model, val_loader, device)
            plot_metrics(metrics)
            plot_differences(trained_model, val_loader, device, config["data"]["classes"])
            model_save_path = os.path.join(config['training']['checkpoint_dir'], f'best_kfold_without_Aug_{fold}.pth')
            torch.save(trained_model.state_dict(), model_save_path)
    else:
        model_instance = model_module.get_model(config["model"], num_classes).to(device)
        train_loader, val_loader, test_loader = create_data_loaders(config['data'])
        trained_model, metrics = train.train_model(model_instance, train_loader, val_loader, config)
        test.test_model(trained_model, test_loader, device)
        plot_metrics(metrics)
        plot_differences(trained_model, test_loader, device, config["data"]["classes"])
        model_save_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pth')
        torch.save(trained_model.state_dict(), model_save_path)
    print("Training completed successfully!")

if __name__ == '__main__':
    main()

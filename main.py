import os
from glob import glob
import random
import numpy as np
import argparse
import re

import torch
import wandb
from config import Config
from data_utils import load_data, prepare_dataloaders
from models import get_model, SVMClassifier, EmotionRecognitionModel_v1, EmotionRecognitionModel_v2
from train_utils import train_model, evaluate_model
from evaluation import compare_models
from visualization import visualize_results
from hyperparameter_search import run_hyperparameter_sweep


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #print(torch.cuda.current_device()) 
        print(f'\n#####GPU verified. {torch.cuda.get_device_name(0)}')
    return device

def prep_model(config, train_loader, is_sweep=False):
    device = set_seed(config.SEED)
    print(f"Using device: {device}")
    config.device = device
    print(f'\n###### Preparing Model...\nCurrent path: {config.CKPT_SAVE_PATH}\n\nModel:{config.MODEL}\nOptimizer:{config.OPTIMIZER}\nActivation: {config.ACTIVATION}\nBatch size: {config.BATCH_SIZE}\nlearning rate: {config.lr}\nDrop out: {config.DROPOUT_RATE}\nNum of epoch: {config.NUM_EPOCHS}\n')
    
    # # Data settings
    # RATIO_TRAIN: float = 0.7
    # RATIO_TEST: float = 0.15
    # DATA_NAME= "RAVDESS_audio_speech"
    # # Paths
    # PROJECT_DIR: str = "Project"#"NMA_Project_SER"
    #### Optimizer & Cost function 
    model = get_model(config, train_loader)
    if config.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    elif config.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config.lr), momentum=0.9)
    else:
        print('err optimizer')
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    criterion = torch.nn.CrossEntropyLoss()
    #### Model loading or start new
    if os.path.exists(config.CKPT_SAVE_PATH):
        if not is_sweep: # load model
            #id_wandb = wandb.util.generate_id()
            
            model, optimizer, initial_epoch, best_val_accuracy, id_wandb = load_checkpoint(config.CKPT_SAVE_PATH, model, optimizer, device)
            print(f"Resuming training from epoch {initial_epoch}. Best val accuracy: {best_val_accuracy:.3f}\nWandb id loaded: {id_wandb}\nWandb project: {config.WANDB_PROJECT}")
            config.id_wandb=id_wandb
            wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, settings=wandb.Settings(start_method="thread"))
        else: 
            print('\n####### Sweep starts. ')
            initial_epoch = 1
            id_wandb = wandb.util.generate_id()
            wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, resume=False, settings=wandb.Settings(start_method="thread"))
            model = get_model(config, train_loader)
    else:
        print('No trained data.')
        initial_epoch = 1
        id_wandb = wandb.util.generate_id()
        print(f'Wandb id generated: {id_wandb}')
        wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)
        model = get_model(config, train_loader)

    config.initial_epoch = initial_epoch
    config.id_wandb = id_wandb
    model = model.to(device)
    return model, optimizer, criterion, device

def load_checkpoint(ckpt_path, model, optimizer, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_accuracy = checkpoint['best_val_accuracy']
    id_wandb = checkpoint['id_wandb']
    return model, optimizer, start_epoch, best_val_accuracy, id_wandb

def read_best_model_info(config):
    info_path = os.path.join(config.MODEL_DIR, 'best_model_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return f.read().strip()
    return None
  
def write_best_model_info(config, info):
    info_path = os.path.join(config.MODEL_DIR, 'best_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(info)

def find_best_model(config, test_loader, device, exclude_models=None):
    model_folders = [f for f in os.listdir(config.MODEL_DIR) if os.path.isdir(os.path.join(config.MODEL_DIR, f))]
    best_models = []
    
    for folder in model_folders:
        folder_path = os.path.join(config.MODEL_DIR, folder)
        best_model_files = glob(os.path.join(folder_path, '*best_model*.pth'))
        best_models.extend(best_model_files)
    
    if exclude_models:
        best_models = [m for m in best_models if m not in exclude_models]
    
    if not best_models:
        print(f"No new best model files found in {config.MODEL_DIR}")
        return None
    
    print(f"Found {len(best_models)} best model files:")
    for model_path in best_models:
        print(model_path)
    
    best_performance = float('inf')  # Using loss as the metric, lower is better
    best_model_path = None
    
    for model_path in best_models:
        try:
            model = get_model(config, test_loader)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            loss, _, _, _, _, _, _ = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(), device)
            
            print(f"Model: {model_path}")
            print(f"Test Loss: {loss:.4f}")
            
            if loss < best_performance:
                best_performance = loss
                best_model_path = model_path
        
        except Exception as e:
            print(f"Error evaluating model {model_path}: {str(e)}")
            continue
    
    if best_model_path:
        print(f"\nBest performing model: {best_model_path}")
        print(f"Best performance (Loss): {best_performance:.4f}")
        
        # Save best model info
        info = f"Best model path: {best_model_path}\nBest performance (Loss): {best_performance:.4f}"
        write_best_model_info(config, info)
    else:
        print("\nNo valid models found or all models failed evaluation.")
    
    return best_model_path


def get_next_version(model_path):
    dir_name, file_name = os.path.split(model_path)
    match = re.search(r'_v(\d+)', file_name)
    if match:
        current_version = int(match.group(1))
        new_version = current_version + 1
        new_file_name = re.sub(r'_v\d+', f'_v{new_version}', file_name)
    else:
        new_file_name = file_name.replace('.pth', '_v2.pth')
    return os.path.join(dir_name, new_file_name)

def list_models(config):
    models = glob(os.path.join(config.MODEL_DIR, '**', '*best_model*.pth'), recursive=True)
    if not models:
        print("No trained models found.")
        return None
    print("Available models:")
    for i, model_path in enumerate(models, 1):
        print(f"{i}. {model_path}")
    return models

def print_menu():
    print("\nEmotion Recognition Model - Choose an option:")
    print("1. Train a new model")
    print("2. Resume training")
    print("3. Run hyperparameter search")
    print("4. Evaluate existing model")
    print("5. Compare with baseline models")
    print("6. Find best performing model")
    print("7. Exit")
    return input("Enter your choice (1-7): ")
def main(args=None):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data, labels = load_data(config)
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, config)

    if args is None:
        while True:
            choice = print_menu()
            if choice == '1':
                if os.path.exists(config.MODEL_SAVE_PATH) or os.path.exists(config.CKPT_SAVE_PATH):
                    user_input = input("Existing trained data exists. Trained with a new version? (y/n): ").lower()
                    if user_input == 'y':
                        config.MODEL_SAVE_PATH = get_next_version(config.MODEL_SAVE_PATH)
                        config.CKPT_SAVE_PATH = get_next_version(config.CKPT_SAVE_PATH)
                        print(f"New model will be saved as {config.MODEL_SAVE_PATH}.")
                        args = argparse.Namespace(mode='train')
                    elif user_input == 'n':
                        args = argparse.Namespace(mode='resume')
                    else:
                        print("Something's wrong. Try again.")
                        continue
                else:
                    args = argparse.Namespace(mode='train')
            elif choice == '2':
                args = argparse.Namespace(mode='resume')
            elif choice == '3':
                args = argparse.Namespace(mode='sweep', sweeps=config.N_SWEEP)
            elif choice == '4':
                args = argparse.Namespace(mode='evaluate')
            elif choice == '5':
                args = argparse.Namespace(mode='benchmark')
            elif choice == '6':
                args = argparse.Namespace(mode='find_best')
            elif choice == '7':
                print("Exit program.")
                return
            else:
                print("Something's wrong. Try again.")
                continue
            break

    if args.mode == 'train':
        config.NUM_EPOCHS = int(input("Number of epoch for training: "))
        print(f"Total {config.NUM_EPOCHS} epochs of training.\n")
        model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=False)
        history, best_val_accuracy = train_model(model, train_loader, val_loader, config, device, optimizer, criterion)
        
        visualize_results(config, model, train_loader, device, history, 'train')
        visualize_results(config, model, val_loader, device, history, 'val')
        visualize_results(config, model, test_loader, device, history, 'test')
        
        print(f"Best val accuracy: {best_val_accuracy:.4f}")
    
    elif args.mode == 'resume':
        models = list_models(config)
        if models:
            print(models)
            model_index = int(input("Select the model to retrain: ")) - 1
            config.MODEL_SAVE_PATH = models[model_index]
            config.CKPT_SAVE_PATH = config.MODEL_SAVE_PATH.replace('best_model', 'checkpoint')
            additional_epochs = int(input("Number of epoch for training: "))
            config.NUM_EPOCHS += additional_epochs
            model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=False)
            history, best_val_accuracy = train_model(model, train_loader, val_loader, config, device, optimizer, criterion)
            visualize_results(config, model, train_loader, device, history, 'train')
            visualize_results(config, model, val_loader, device, history, 'val')
            visualize_results(config, model, test_loader, device, history, 'test')
            print(f"Best val accuracy: {best_val_accuracy:.4f}")
        else:
            print("No models to resume.")

    elif args.mode == 'sweep':
        run_hyperparameter_sweep(config, data, labels)
    
    elif args.mode == 'evaluate':
        model, _, criterion, device = prep_model(config, train_loader, is_sweep=False)
        test_loss, test_metrics = evaluate_model(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")
        visualize_results(config, model, test_loader, device, None, 'test')
    
    elif args.mode == 'benchmark':
        model, _, criterion, device = prep_model(config, test_loader, is_sweep=False)
        compare_models(model, train_loader, val_loader, test_loader, config, device)
    
    elif args.mode == 'find_best':
        best_model_path = find_best_model(config, test_loader, device)
        if best_model_path:
            print(f"Best model found: {best_model_path}")
        else:
            print("No valid model found.")
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emotion Recognition Model")
    parser.add_argument("--mode", choices=['train', 'sweep', 'evaluate', 'benchmark', 'find_best'],
                        help="Mode of operation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--sweeps", type=int, default=10, help="Number of sweeps for hyperparameter search")
    args = parser.parse_args()
    
    if args.mode:
        main(args)
    else:
        main()

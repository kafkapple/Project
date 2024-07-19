import os
from glob import glob
import argparse
import re
import torch
from config import Config
from data_utils import load_data, prepare_dataloaders
from models import prep_model, get_model, SVMClassifier, EmotionRecognitionModel_v1, EmotionRecognitionModel_v2
from train_utils import train_model, evaluate_model, load_checkpoint
from evaluation import compare_models
from visualization import visualize_results
from hyperparameter_search import run_hyperparameter_sweep
import wandb

def read_best_model_info(config):
    info_path = os.path.join(config.MODEL_BASE_DIR, 'best_model_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return f.read().strip()
    return None
  
def write_best_model_info(config, info):
    info_path = os.path.join(config.MODEL_BASE_DIR, 'best_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(info)

def find_best_model(config, test_loader, device, exclude_models=None):
    model_folders = [f for f in os.listdir(config.MODEL_BASE_DIR) if os.path.isdir(os.path.join(config.MODEL_BASE_DIR, f))]
    best_models = []
    
    for folder in model_folders:
        folder_path = os.path.join(config.MODEL_BASE_DIR, folder)
        best_model_files = glob(os.path.join(folder_path, '*best_model*.pth'))
        best_models.extend(best_model_files)
    
    if exclude_models:
        best_models = [m for m in best_models if m not in exclude_models]
    
    if not best_models:
        print(f"No new best model files found in {config.MODEL_BASE_DIR}")
        return None
    
    print(f"Found {len(best_models)} putative best model files:")
    for model_path in best_models:
        print(model_path)
    
    best_performance = float('inf')  # Using loss as the metric, lower is better
    best_model_path = None
    dict_models={}
    for model_path in best_models:
        
        try:
            model = get_model(config, test_loader)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            loss, accuracy, _, _, f1, _, _ = evaluate_model(model, test_loader, torch.nn.CrossEntropyLoss(), device)
            dict_models[model_path]={'loss':loss, 'accuracy':accuracy,'f1':f1}
            
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
        print(f"Other metrics\n{dict_models[best_model_path]}")
        
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
    models = glob(os.path.join(config.MODEL_BASE_DIR, '**', '*best_model*.pth'), recursive=True)
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
    # Best model i
    best_model_info_path = os.path.join(config.MODEL_BASE_DIR, 'best_model_info.txt')
    if os.path.exists(best_model_info_path):
        print(f'\n##### Trained models are found #####\n')
        with open(best_model_info_path, 'r') as f:
            best_model_info = f.read().strip()
        
        best_model_path = None
        for line in best_model_info.split('\n'):
            if line.startswith("Best model path:"):
                best_model_path = line.split(": ", 1)[1].strip()
                break
        
        if best_model_path:
            config.MODEL_SAVE_PATH = best_model_path
            config.CKPT_SAVE_PATH = best_model_path.replace("best_model", "checkpoint")
            print(f"Current best model: {config.MODEL_SAVE_PATH}")
        else:
            print("Best model information found, but path is not available.")
    else:
        print("No best model information found.")
    
    ### Dataset
    
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
                        config.id_wandb = wandb.util.generate_id()
                    elif user_input == 'n':
                        args = argparse.Namespace(mode='resume')
                    else:
                        print("Something's wrong. Try again.")
                        continue
                else:
                    args = argparse.Namespace(mode='train')
                    config.id_wandb = wandb.util.generate_id()
            elif choice == '2':
                args = argparse.Namespace(mode='resume')
            elif choice == '3':
                args = argparse.Namespace(mode='sweep', sweeps=config.N_SWEEP)
                #config.id_wandb = wandb.util.generate_id()
            elif choice == '4':
                args = argparse.Namespace(mode='evaluate')
                config.id_wandb = wandb.util.generate_id()
            elif choice == '5':
                args = argparse.Namespace(mode='benchmark')
                config.id_wandb = wandb.util.generate_id()
            elif choice == '6':
                args = argparse.Namespace(mode='find_best')
                config.id_wandb = wandb.util.generate_id()
            elif choice == '7':
                print("Exit program.")
                return
            else:
                print("Something's wrong. Try again.")
                continue
            break
    config.CUR_MODE=args.mode
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
            model_index = int(input("Select the model to retrain: ")) - 1
            config.MODEL_SAVE_PATH = models[model_index]
            config.CKPT_SAVE_PATH = config.MODEL_SAVE_PATH.replace('best_model', 'checkpoint')
            additional_epochs = int(input("Number of epoch for training: "))
            print(f'Model will be trained for {additional_epochs} epochs')
            
            model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=False)
            _, _, start_epoch, _, _ = load_checkpoint(config.CKPT_SAVE_PATH, model, optimizer, device)
            
            config.initial_epoch = start_epoch
            config.NUM_EPOCHS = additional_epochs
            
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

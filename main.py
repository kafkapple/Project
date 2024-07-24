import os
from glob import glob
import argparse
import re
import torch
import torch.nn as nn
from config import Config
from data_utils import load_data, prepare_dataloaders, prep_audio, preprocess_data, preprocess_data_meld
from models import list_models, chk_best_model_info, find_best_model, prep_model
from train_utils import train_model, evaluate_model, load_checkpoint
from evaluation import compare_models
from visualization import visualize_results
from hyperparameter_search import run_hyperparameter_sweep
import pandas as pd
def generate_unique_filename(filename):
    name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(new_filename):
        new_filename = f"{name}_v{counter}{ext}"
        counter += 1
    
    return new_filename

def init_weights(m):
    if isinstance(m, nn.Linear):
        #nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0.01)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


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



def print_menu():
    print("\n<<< NMA 2024 Emotion Recognition Model >>> - Choose an option:")
    print("0. Download and Prepare Dataset")
    print("1. Train a new model")
    print("2. Resume training")
    print("3. Run hyperparameter search")
    print("4. Evaluate existing model")
    print("5. Compare with baseline models")
    print("6. Find best performing model")
    print("7. Exit")
    return input("Enter your choice (0-7): ")
def main(args=None):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args is None:
        while True:
            choice = print_menu()
            if choice == '0':
                args = argparse.Namespace(mode='prep_data')
            elif choice == '1':
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
    
    # Dataset
    # Best model info chk
    chk_best_model_info(config)
    
    config.CUR_MODE=args.mode
    config.WANDB_PROJECT = args.mode+'_'+config.MODEL+'_'+config.DATA_NAME
    
    file_name, _ = os.path.splitext(os.path.basename(config.MODEL_SAVE_PATH))
    file_name = file_name.replace('best_model_', '')
    config.model_name=file_name
    print('Model name: ',config.model_name)
    
    folder_path = os.path.dirname(config.MODEL_SAVE_PATH)
    new_path=os.path.join(folder_path, config.model_name) # chk
    config.MODEL_DIR=new_path
    print('Model path New: ', config.MODEL_DIR)

    print('epoch: ',config.global_epoch)
        ### Dataset
    if args.mode =='prep_data':
        SELECT_DATA = input("Select dataset type\n1. Audio dataset (RAVDESS Speech)\n2. Multi-modal dataset (MELD)\n")
        if SELECT_DATA =='1':
            config.DATA_NAME = 'RAVDESS'
        elif SELECT_DATA =='2':
            config.DATA_NAME ='MELD'
            config.TARGET=input("train or test dataset?\n")
    
            print(f'Dataset: {config.DATA_NAME} / {config.TARGET} will be prepared.\nSamples: {config.N_SAMPLE}')
   
        data_dir= load_data(config) 

        if SELECT_DATA =='2':
            label_info_df = pd.read_csv(f'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/{config.TARGET}_sent_emo.csv')
            label_info_df = label_info_df.sample(config.N_SAMPLE, random_state=config.SEED, ignore_index=True)
            data_meld_path=os.path.join(config.extracted_path, f'{config.TARGET}_audio')
            if not os.path.exists(data_meld_path):
                print('Audio extraction is needed.')
                prep_audio(config, label_info_df, data_meld_path, config.TARGET)
            else:
                print('Audio extraction from MELD data is done.')
            data, labels = preprocess_data_meld(data_meld_path, label_info_df)
        else:
            data, labels = preprocess_data(data_dir)
        train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, config)
        
    elif args.mode == 'train':
        IS_RESUME=False
        select_data = int(input('Select dataset for training.\n1. RAVDESS\n2. MELD\n3. MELD toy\n'))
        if select_data ==1:
            config.DATA_NAME='RAVDESS'
        elif select_data == 2:
            config.DATA_NAME='MELD'
        elif select_data == 3:
            config.DATA_NAME = 'MELD_toy'
        else:
            print('ERR')
        select_model = int(input('Select Model type.\n1. Classifier only\n2. Pretrained model \n3. Finetuned model\n'))
        if select_model ==1:
            config.MODEL ="classifier_only"#"wav2vec_v2"  "wav2vec_finetuned"
            config.BOOL_MODEL_INIT =True
        elif select_model == 2:
            config.MODEL="wav2vec_pretrained"#"wav2vec_v2" "classifer" "wav2vec_finetuned"
        elif select_model == 3:
            config.MODEL = "wav2vec_finetuned"
        else:
            print('ERR')
            
        config.update_path()
         #data_dir = load_data(config) 
        data_dir = os.path.join(config.DATA_DIR, config.DATA_NAME)
        print('Data Dir: ', data_dir)
        #extracted_path = os.path.join(config.DATA_DIR, f"{config.DATA_NAME}.Raw")
        if config.DATA_NAME=="MELD":
            data_dir=os.path.join(data_dir, 'train_audio')
            
            text_train_df= pd.read_csv(os.path.join(config.DATA_DIR, 'MELD_train_sampled.csv'))
            data, labels = preprocess_data_meld(data_dir, text_train_df)
            dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
            labels=[dict_label[val] for val in labels]
            config.LABELS_EMOTION =config.LABELS_EMO_MELD
        elif config.DATA_NAME=="MELD_toy":
            data_dir=os.path.join(data_dir, 'train_audio_toy')

            text_train_df= pd.read_csv(os.path.join(config.DATA_DIR, 'MELD_train_sampled_toy.csv'))
            data, labels = preprocess_data_meld(data_dir, text_train_df)
            dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
            labels=[dict_label[val] for val in labels]
            config.LABELS_EMOTION =config.LABELS_EMO_MELD
        elif config.DATA_NAME=='RAVDESS':
            data, labels = preprocess_data(data_dir)
        else:
            print('err data')
        
        train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, config)
        
        
        if os.path.exists(config.MODEL_SAVE_PATH):
            config.MODEL_SAVE_PATH=generate_unique_filename(config.MODEL_SAVE_PATH)
            config.CKPT_SAVE_PATH=generate_unique_filename(config.CKPT_SAVE_PATH)
            print(f'New model will be trained: {config.MODEL_SAVE_PATH}')
            file_name, _ = os.path.splitext(os.path.basename(config.MODEL_SAVE_PATH))
            
            file_name = file_name.replace('best_model_', '')
            
            os.makedirs(new_path, exist_ok=True)
            os.makedirs(os.path.join(new_path, 'results'), exist_ok=True)
            print(new_path)

            
        config.NUM_EPOCHS = int(input("Number of epoch for training: "))
        
        model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=False)
        
        #### !! epoch +1 but not wanted?! -> model prep problem
        print('Model initialization...')
        if config.MODEL_INIT:
            model.apply(init_weights)   
        history, best_val_loss, best_val_acc = train_model(model, train_loader, val_loader, config, device, optimizer, criterion)
        config.history=history
        
        visualize_results(config, model, test_loader, device, history, 'test')
        
        print(f"Best val loss: {best_val_loss:.4f}")
    elif args.mode == 'resume':
        select_data = int(input('Select dataset for training.\n1. RAVDESS\n2. MELD\n'))
        if select_data ==1:
            config.DATA_NAME='RAVDESS'
        elif select_data == 2:
            config.DATA_NAME='MELD'
        else:
            print('ERR')
        
         #data_dir = load_data(config) 
        
         #data_dir = load_data(config) 
        data_dir = os.path.join(config.DATA_DIR, config.DATA_NAME)
        #extracted_path = os.path.join(config.DATA_DIR, f"{config.DATA_NAME}.Raw")
        if config.DATA_NAME=="MELD":
            text_train_df= pd.read_csv(os.path.join(config.DATA_DIR, 'MELD_train_sampled.csv'))
            data, labels = preprocess_data_meld(os.path.join(data_dir, 'train_audio'), text_train_df)
            dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
            labels=[dict_label[val] for val in labels]
            config.LABELS_EMOTION =config.LABELS_EMO_MELD
        elif config.DATA_NAME=="MELD_toy":
            text_train_df= pd.read_csv(os.path.join(config.DATA_DIR, 'MELD_train_sampled_toy.csv'))
            data, labels = preprocess_data_meld(os.path.join(data_dir, 'train_audio_toy'), text_train_df)
            dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
            labels=[dict_label[val] for val in labels]
            config.LABELS_EMOTION =config.LABELS_EMO_MELD
            
        elif config.DATA_NAME=='RAVDESS':
            data, labels = preprocess_data(data_dir)
        else:
            print('err data')
        
        train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, config)
        
        config.WANDB_PROJECT = 'train'+'_'+config.MODEL+'_'+config.DATA_NAME
        config.IS_RESUME=True
        # models = list_models(config)
        # if models:
        #     model_index = int(input("Select the model to retrain: ")) - 1
        #     config.MODEL_SAVE_PATH = models[model_index]
        #     config.CKPT_SAVE_PATH = config.MODEL_SAVE_PATH.replace('best_model', 'checkpoint')
        #     file_name, _ = os.path.splitext(os.path.basename(config.MODEL_SAVE_PATH))
        #     file_name = file_name.replace('best_model_', '')
        #     config.model_name=file_name
        #     folder_path = os.path.dirname(config.MODEL_SAVE_PATH)
        #     new_path=os.path.join(folder_path, config.model_name)
        #     config.MODEL_DIR=new_path
        #     os.makedirs(new_path, exist_ok=True)
        #     os.makedirs(os.path.join(new_path, 'results'), exist_ok=True)
        #     print(new_path)
            
        #     additional_epochs = int(input("Number of epoch for training: "))
        #     print(f'Model will be trained for {additional_epochs} epochs')
            
        #     model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=False)
        #     _,_, global_epoch, best_val_loss,_ = load_checkpoint(config, model, optimizer, device)
            
         
        #     config.NUM_EPOCHS = additional_epochs
        #     print(f'Preparing resume training. Global epoch is set to previous epoch +1: {config.global_epoch}')
            
        #     history, best_val_loss, best_val_acc = train_model(model, train_loader, val_loader, config, device, optimizer, criterion)
        #     config.history=history
        #     visualize_results(config, model, test_loader, device, history, 'test')
        #     print(f"Best val loss: {best_val_loss:.4f}")
        # else:
        #     print("No models to resume.")

    elif args.mode == 'sweep':
        config.IS_SWEEP=True
        num_epoch_sweep = input("Number of epoch / sweep for Hyperparameter Search: (e.g., 10 5 # 10 epoch / 5 sweep)")
        config.NUM_EPOCHS, config.N_SWEEP = [int(i) for i in num_epoch_sweep.split(' ')]
        print(f'{config.NUM_EPOCHS}-epoch / {config.N_SWEEP}-sweep\n')
        run_hyperparameter_sweep(config, train_loader, val_loader)
    
    elif args.mode == 'evaluate':
        model, _, criterion, device = prep_model(config, train_loader, is_sweep=False)
        #test_loss, test_metrics 
        test_loss, test_accuracy, _, _, test_f1, _, _ = evaluate_model(config, model, test_loader, criterion, device)
        
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
        visualize_results(config, model, test_loader, device, None, 'test')
    
    elif args.mode == 'benchmark':
        config.SWEEP_NAIVE=True

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

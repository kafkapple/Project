import os
import torch
import torch.nn as nn
from sklearn.svm import SVC
import joblib
import wandb
from train_utils import load_checkpoint
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from glob import glob

from train_utils import evaluate_model
from transformers import Wav2Vec2Model
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
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

def list_models(config):
    models = glob(os.path.join(config.MODEL_BASE_DIR, '**', '*best_model*.pth'), recursive=True)
    if not models:
        print("No trained models found.")
        return None
    print("Available models:")
    for i, model_path in enumerate(models, 1):
        print(f"{i}. {model_path}")
    return models

def chk_best_model_info(config):
    if os.path.exists(config.best_model_info_path):
        print(f'\n##### Trained model info exists #####\n')
        with open(config.best_model_info_path, 'r') as f:
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


def read_best_model_info(config):
    info_path = os.path.join(config.MODEL_BASE_DIR, 'best_model_info.txt')
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            return f.read().strip()
    return None
  
def write_best_model_info(config, info, dict_models):
    info_path = os.path.join(config.MODEL_BASE_DIR, 'best_model_info.txt')
    with open(info_path, 'w') as f:
        f.write(info)
        for i in dict_models:
            f.write(f'Model: {i} / {dict_models[i]}')
            
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
            
            loss, accuracy, _, _, f1, _, _ = evaluate_model(config, model, test_loader, torch.nn.CrossEntropyLoss(), device)
            
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
        write_best_model_info(config, info, dict_models)
    else:
        print("\nNo valid models found or all models failed evaluation.")
    
    return best_model_path

def prep_model(config, train_loader, is_sweep=False):
    device = set_seed(config.SEED)
    print(f"Using device: {device}")
    config.device = device
    print(f'\n###### Preparing Model...\nCurrent path: {config.CKPT_SAVE_PATH}\n\nModel:{config.MODEL}\nOptimizer:{config.OPTIMIZER}\nActivation: {config.ACTIVATION}\nBatch size: {config.BATCH_SIZE}\nlearning rate: {config.lr}\nDrop out: {config.DROPOUT_RATE}\n')
    
    #### Optimizer & Cost function 
    model = get_model(config, train_loader)
    
    if config.OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=config.weight_decay, lr=float(config.lr))
    elif config.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(config.lr), momentum=0.9)
    else:
        print('err optimizer')
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")
    
    criterion = torch.nn.CrossEntropyLoss()
    #### Model loading or start new
    if config.CUR_MODE == 'train' or config.CUR_MODE =='benchmark':
        print('New training / benchmark begins.')
        global_epoch = 0
        id_wandb = wandb.util.generate_id()
        print(f'Wandb id generated: {id_wandb}')
        config.id_wandb = id_wandb
        wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)#, resume=True)
        # model = get_model(config, train_loader)
    elif config.CUR_MODE == 'resume':
        
        model, optimizer, global_epoch, best_val_accuracy, id_wandb= load_checkpoint(config, model, optimizer, device)
   
        print(f"Resuming training from epoch {global_epoch}. Best val accuracy: {best_val_accuracy:.3f}\nWandb id loaded: {config.id_wandb}\nWandb project: {config.WANDB_PROJECT}")
        
        config.id_wandb=id_wandb
        wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, resume="must", settings=wandb.Settings(start_method="thread"))
    elif config.CUR_MODE == 'sweep':
        print('\n####### Sweep starts. ')
        global_epoch = 0
        #id_wandb = wandb.util.generate_id()
        #config.WANDB_PROJECT+="_"+config.CUR_MODE
        id_wandb=config.id_wandb
        wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, resume=False, settings=wandb.Settings(start_method="thread"))
        model = get_model(config, train_loader)

    else:
        print('\n\n######### to be checked #####\n\n')
        global_epoch = 0
        id_wandb = wandb.util.generate_id()
        print(f'Wandb id generated: {id_wandb}')
        config.id_wandb = id_wandb
        
        wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)
        
    # if os.path.exists(config.CKPT_SAVE_PATH) and config.IS_RESUME:
    #     model, optimizer, global_epoch, best_val_accuracy, id_wandb= load_checkpoint(config, model, optimizer, device)
    #     if not is_sweep: # load model
    #         #id_wandb = wandb.util.generate_id()
    #         print(f"Resuming training from epoch {global_epoch}. Best val accuracy: {best_val_accuracy:.3f}\nWandb id loaded: {config.id_wandb}\nWandb project: {config.WANDB_PROJECT}")
    #         if config.CUR_MODE == 'benchmark':
    #             #config.WANDB_PROJECT+= "_"+config.CUR_MODE
    #             print(f'But this is for benchmark. New wandb id is generated: {id_wandb}\nWandb project: {config.WANDB_PROJECT}')
              
    #         config.id_wandb=id_wandb
    #         wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, resum=True, settings=wandb.Settings(start_method="thread"))
    #     else: 
    #         print('\n####### Sweep starts. ')
    #         global_epoch = 1
    #         #id_wandb = wandb.util.generate_id()
    #         #config.WANDB_PROJECT+="_"+config.CUR_MODE
    #         id_wandb=config.id_wandb
    #         wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, resume=False, settings=wandb.Settings(start_method="thread"))
    #         model = get_model(config, train_loader)
    # else:
    #     print('No trained data was given. New training begins.')
    #     global_epoch = 1
    #     id_wandb = wandb.util.generate_id()
    #     print(f'Wandb id generated: {id_wandb}')
    #     config.id_wandb = id_wandb
        
    #     wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS)#, resume=True)
    #     model = get_model(config, train_loader)
    config.global_epoch = global_epoch

    model = model.to(device)
    return model, optimizer, criterion, device

def get_model(config, train_loader):
    
    if config.MODEL == 'classifier_only':
        print('classifer only')
        model = EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_loader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION, use_wav2vec=False)
        
    elif config.MODEL =='wav2vec_pretrained':
        print('Pretrained model loaded.')
        
        model= EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_loader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION, use_wav2vec=True)
    elif config.MODEL =='wav2vec_finetuned':
        
        config.path_pretrained= os.path.join(config.MODEL_BASE_DIR, 'wav2vec_I_fine_tune_best')
        print('Finetuned model loaded from: ',config.path_pretrained )
        model= EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_loader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION, use_wav2vec=True)
    
    # elif config.MODEL == 'SVM_C':
    #     return SVMClassifier(train_loader.dataset[0][0].shape[1], num_classes=len(config.LABELS_EMOTION))
    else:
        raise ValueError(f"Unknown model type: {config.MODEL}")
    
    return model#model_class(
  
# ### Models
# class SVMClassifier:
#     def __init__(self, input_size, num_classes):
#         self.model = SVC(kernel='rbf')
#         self.input_size=input_size
#         self.num_classes=num_classes

#     def fit(self, X, y):
#         self.model.fit(X, y)

#     def predict(self, X):
#         return self.model.predict(X)

#     def save(self, path):
#         joblib.dump(self.model, path)

#     def load(self, path):
#         self.model = joblib.load(path)
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class EmotionRecognitionBase(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = self._get_activation(activation)
        self.input_size = input_size
        self.num_classes = num_classes
    
    def _get_activation(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

class EmotionRecognitionModel_v2(EmotionRecognitionBase):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__(input_size, num_classes, dropout_rate, activation)
        momentum = 0.01
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=momentum)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, momentum=momentum)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64, momentum=momentum)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        #x = x.squeeze(1)  # Squeeze the input to remove dimensions of size 1
        x = x.view(x.size(0), -1)  # Flatten the input if necessary
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

class EmotionRecognitionWithWav2Vec(nn.Module):
    def __init__(self, num_classes, config, dropout_rate=0.5, activation='relu', use_wav2vec=True, input_size=None):
        super().__init__()
        
        self.use_wav2vec = use_wav2vec
        self.config = config
        
        if use_wav2vec:
            # wav2vec 모델 로드
            self.wav2vec = Wav2Vec2Model.from_pretrained(self.config.path_pretrained)#"facebook/wav2vec2-base")
            self.wav2vec.config.mask_time_length = config.mask_time_length
            wav2vec_output_size = self.wav2vec.config.hidden_size
        else:
            # wav2vec을 사용하지 않을 경우의 입력 크기 (예: MFCC 특징의 크기)
            wav2vec_output_size = input_size#self.wav2vec.config.hidden_size#40  # 예시 값, 실제 입력 크기에 맞게 조정 필요
        
        # EmotionRecognitionModel_v2 인스턴스화
        self.emotion_classifier = EmotionRecognitionModel_v2(
            input_size=wav2vec_output_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
    def forward(self, input_values):
        if self.use_wav2vec:
            if self.config.VISUALIZE:
                print(f"Input shape before processing: {input_values.shape}")
            # 입력 데이터 형태 조정
            if input_values.dim() == 4:
                input_values = input_values.squeeze(2)  # [batch, 1, 1, sequence] -> [batch, 1, sequence]
            if input_values.dim() == 3:
                input_values = input_values.squeeze(1)  # [batch, 1, sequence] -> [batch, sequence]
            # wav2vec 특징 추출
            if self.config.VISUALIZE:
                print(f"Input shape after processing: {input_values.shape}")
            wav2vec_outputs = self.wav2vec(input_values).last_hidden_state
            # 시퀀스의 평균을 취하여 고정 크기 벡터 얻기
            features = torch.mean(wav2vec_outputs, dim=1)
        else:
            # wav2vec을 사용하지 않을 경우, 입력을 그대로 사용
            #features = input_values
            features = input_values.view(input_values.size(0), -1)  # Flatten the input
        
        # 감정 분류
        emotion_logits = self.emotion_classifier(features)
        
        return emotion_logits

# 모델 초기화 예시


# 모델 초기화 예시

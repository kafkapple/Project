import os
import torch
import torch.nn as nn
from sklearn.svm import SVC
import joblib
import wandb
from train_utils import load_checkpoint
import numpy as np
import random

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
    print(f'\n###### Preparing Model...\nCurrent path: {config.CKPT_SAVE_PATH}\n\nModel:{config.MODEL}\nOptimizer:{config.OPTIMIZER}\nActivation: {config.ACTIVATION}\nBatch size: {config.BATCH_SIZE}\nlearning rate: {config.lr}\nDrop out: {config.DROPOUT_RATE}\n')
    
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
            if config.CUR_MODE == 'benchmark':
                id_wandb = wandb.util.generate_id()
                config.WANDB_PROJECT+= "_"+config.CUR_MODE
                print(f'But this is for benchmark. New wandb id is generated: {id_wandb}\nWandb project: {config.WANDB_PROJECT}')
              
            config.id_wandb=id_wandb
            wandb.init(id=id_wandb, project=config.WANDB_PROJECT, config=config.CONFIG_DEFAULTS, settings=wandb.Settings(start_method="thread"))
        else: 
            print('\n####### Sweep starts. ')
            initial_epoch = 1
            id_wandb = wandb.util.generate_id()
            config.WANDB_PROJECT+="_"+config.CUR_MODE
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

def get_model(config, train_loader):
    if config.MODEL == 'wav2vec_v1':
        model_class = EmotionRecognitionModel_v1
    elif config.MODEL == 'wav2vec_v2': 
        model_class = EmotionRecognitionModel_v2
    elif config.MODEL == 'SVM_C':
        return SVMClassifier(train_loader.dataset[0][0].shape[1], num_classes=len(config.LABELS_EMOTION))
    else:
        raise ValueError(f"Unknown model type: {config.MODEL}")
    
    return model_class(
        input_size=train_loader.dataset[0][0].shape[1],
        num_classes=len(config.LABELS_EMOTION),
        dropout_rate=config.DROPOUT_RATE,
        activation=config.ACTIVATION
    )
class SVMClassifier:
    def __init__(self, input_size, num_classes):
        self.model = SVC(kernel='rbf')

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        
class EmotionRecognitionBase(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = self._get_activation(activation)
        
    def _get_activation(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

class EmotionRecognitionModel_v1(EmotionRecognitionBase):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__(input_size, num_classes, dropout_rate, activation)
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != self.input_size:
            print('x dim = 3')
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            print('x dim = 2')
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of input: {x.shape}")
        x = self.activation(self.conv1(x)) 
        x = self.activation(self.conv2(x))
        x = self.pool(x).squeeze(2)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionRecognitionModel_v2(EmotionRecognitionBase):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__(input_size, num_classes, dropout_rate, activation)
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = x.squeeze(1)  # Squeeze the input to remove dimensions of size 1
        # print(f"Shape after squeeze: {x.shape}")
        x = self.activation(self.bn1(self.fc1(x)))
        # print(f"After fc1 and bn1: {x.shape}")
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        # print(f"After fc2 and bn2: {x.shape}")
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        # print(f"After fc3 and bn3: {x.shape}")
        x = self.dropout(x)
        x = self.fc4(x)
        # print(f"Output shape: {x.shape}")
        return x
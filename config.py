from dataclasses import dataclass, field
import os
import sys
import datetime
from collections import namedtuple


# dropout
# bath norm
# optimizer adam
# activation relu
# early stop 5 epoch (loss)

# scheduler at train_utils CosineAnnealingLR
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
@dataclass
class Config:
    model_name: str = ''
    early_stop_epoch: int = 10
    CUR_MODE: str ='' # current mode
    N_STEP_FIG: int = 2
    # General settings
    SEED: int = 2024
    NUM_EPOCHS: int = 5
    global_epoch:int = 0

    BATCH_SIZE: int = 32
    
    DROPOUT_RATE: float = 0.4
    lr: float = 0.0005
    eta_min: float = 1e-4 
    weight_decay: float =1e-5
    
    # Model settings
    MODEL: str = "wav2vec_v2"
    ACTIVATION: str = "relu"
    OPTIMIZER: str = "adam"
    
    SCHEDULER: bool = True
    GRADIENT_CLIP: bool = True
    
    history: dict = field(default_factory=lambda: {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    })
    N_EMBEDDINGS: int = 500 # n of embeddings to show
    # Data settings
    RATIO_TRAIN: float = 0.7
    RATIO_TEST: float = 0.15
    DATA_NAME= "RAVDESS_audio_speech"
    LABELS_EMOTION: dict = field(default_factory=lambda: {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    })
    LABELS_EMOTION_TEXT: dict = field(default_factory=lambda:{
        'sadness': 0, 'anger': 1, 'love': 2, 'surprise': 3, 'fear': 4, 'joy': 5
        })
    # Paths
    PROJECT_DIR: str = "Project"#"NMA_Project_SER"
    BASE_DIR: str = field(init=False)
    DATA_DIR: str = field(init=False)
    MODEL_DIR: str = field(init=False)
    MODEL_SAVE_PATH: str = field(init=False)
    CKPT_SAVE_PATH: str = field(init=False)
    
    # Metric
    EvaluationResult = namedtuple('EvaluationResult', ['loss', 'accuracy', 'precision', 'recall', 'f1', 'labels', 'predictions'])
    METRIC_AVG='weighted'
    
    # Wandb settings
    IS_RESUME: bool = False
    WANDB_PROJECT: str = field(init=False)
    ENTITY: str = "biasdrive-neuromatch"
    id_wandb: str = ""
    sweep_id: str =""
    IS_SWEEP: bool = False
    SWEEP_NAIVE: bool =True
    N_SWEEP: int = 50
    
    model_benchmark='svm'
    C_val: float = 0.1
    max_iter: int = 1000 # for multi logistic reg
    solver: str = 'saga'
    penalty: str = 'l1'
    kernel: str = 'rbf'
    
    sweep_config = {
        'method': 'bayes',
        'metric': {'goal': 'maximize', 'name': 'val.loss' },
        'parameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.01},
            'batch_size': {'values': [16, 32, 64, 128]},
           # 'num_epochs': {'min': 5, 'max': 50},
            'dropout_rate': {'min': 0.1, 'max': 0.6},
            "activation":{"values":['relu', 'leaky_relu', 'gelu']}
        }
    }
    
    CONFIG_DEFAULTS = {
    "resume":"allow",
    "architecture": f"{MODEL}",
    "dataset": f"{DATA_NAME}",
    #"batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    "global_epoch": global_epoch,
    "batch_size": BATCH_SIZE,
    "model": MODEL,
    "lr": lr,
    "dropout_rate": DROPOUT_RATE,
    "activation": ACTIVATION,
    "optimizer": OPTIMIZER
    }  
    def __post_init__(self):
        current_date = datetime.datetime.now()
        date_str = current_date.strftime("%Y%m%d")
        
        # Get current script path
        project_dir= os.path.dirname(os.path.abspath(__file__))

        #project_dir = os.path.dirname(current_dir)
        os.chdir(project_dir)

        # add project path to Python module seasrch path Project 
        sys.path.insert(0, project_dir)
        
        self.BASE_DIR = project_dir#os.path.join(os.path.dirname(os.getcwd()), self.PROJECT_DIR)
        
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        
        self.MODEL_BASE_DIR=os.path.join(self.BASE_DIR, 'models')
        self.MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, f"{self.MODEL}")
        
        self.MODEL_SAVE_PATH = os.path.join(self.MODEL_DIR, f'best_model_{self.MODEL}.pth')
        self.CKPT_SAVE_PATH = os.path.join(self.MODEL_DIR, f'checkpoint_{self.MODEL}.pth')
        self.best_model_info_path = os.path.join(self.MODEL_BASE_DIR, 'best_model_info.txt')
        #self.WANDB_PROJECT = f"{self.PROJECT_DIR}_{self.MODEL}"#_{date_str}"

        print(f'\n\n##### Current Project Location #####\n-Base Directory: {self.BASE_DIR}\n-Data: {self.DATA_DIR}\n-Models: {self.MODEL_BASE_DIR}-Current Model: {self.MODEL_DIR}\n-Current Model name: {self.MODEL_SAVE_PATH}\n\nFigure will be saved per {self.N_STEP_FIG}-step\n')
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.MODEL_BASE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        
    # def __setattr__(self, name:str, value: any) -> None:
    #     if hasattr(self, name):
    #         old_value = getattr(self, name)
    #         if old_value != value:
    #             print(f"Updating {name}: {old_value} -> {value}")
    #     super().__setattr__(name, value)
    def print_variables(self):
        for var, value in self.__dict__.items():
            if not var.startswith("_"):  # 
                print(f"{var}: {value}")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: {key} is not a valid config variable")

#config = Config()
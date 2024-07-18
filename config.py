from dataclasses import dataclass, field
import os
import sys
import datetime

@dataclass
class Config:
    CUR_MODE: str ='' # current mode
    # General settings
    SEED: int = 2024
    NUM_EPOCHS: int = 5
    N_SWEEP: int = 50
    BATCH_SIZE: int = 32
    lr: float = 0.0005
    DROPOUT_RATE: float = 0.4
    
    # Model settings
    MODEL: str = "wav2vec_v2"
    ACTIVATION: str = "relu"
    OPTIMIZER: str = "adam"
    
    # Data settings
    RATIO_TRAIN: float = 0.7
    RATIO_TEST: float = 0.15
    DATA_NAME= "RAVDESS_audio_speech"
    LABELS_EMOTION: dict = field(default_factory=lambda: {
        0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad',
        4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'
    })
    # Paths
    PROJECT_DIR: str = "Project"#"NMA_Project_SER"
    BASE_DIR: str = field(init=False)
    DATA_DIR: str = field(init=False)
    MODEL_DIR: str = field(init=False)
    MODEL_SAVE_PATH: str = field(init=False)
    CKPT_SAVE_PATH: str = field(init=False)
    
    # Wandb settings
    WANDB_PROJECT: str = field(init=False)
    ENTITY: str = "biasdrive-neuromatch"
    id_wandb: str = ""
    
    
    CONFIG_SWEEP = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "val.loss"},
        "parameters": {
            "BATCH_SIZE": {"values": [16, 32, 64]},
            "MODEL":{"values":['SVM_C','wav2vec_v2']},
            "lr": {"values": [0.0001, 0.0005, 0.001, 0.005, 0.01]},
           # "DROPOUT_RATE": {"values": [0.3, 0.4, 0.5]},
            #"activation":{"values":['relu', 'leaky_relu', 'gelu']},
            "OPTIMIZER":{"values":['adam', 'SGD']}
        },
    }
    CONFIG_DEFAULTS = {
    "resume":"allow",
    "architecture": f"{MODEL}",
    "dataset": f"{DATA_NAME}",
    #"batch_size": BATCH_SIZE,
    "epochs": NUM_EPOCHS,
    # "initial_epoch": initial_epoch,
    "BATCH_SIZE": BATCH_SIZE,
    "MODEL": MODEL,
    "lr": lr,
    "DROPOUT_RATE": DROPOUT_RATE,
    "ACTIVATION": ACTIVATION,
    "OPTIMIZER": OPTIMIZER
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
        self.MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, f"{self.MODEL}_v1_{date_str}")
        
        self.MODEL_SAVE_PATH = os.path.join(self.MODEL_DIR, f'best_model_{self.MODEL}.pth')
        self.CKPT_SAVE_PATH = os.path.join(self.MODEL_DIR, f'checkpoint_{self.MODEL}.pth')
        
        self.WANDB_PROJECT = f"{self.PROJECT_DIR}_{self.MODEL}_{date_str}"
        
        print(f'\n\n##### Current Project Location #####\n-Base Directory: {self.BASE_DIR}\n-Data: {self.DATA_DIR}\n-Models: {self.MODEL_BASE_DIR}\n-Current Model: {self.MODEL_DIR}\n-Current Model name: {self.MODEL_SAVE_PATH}\n\n')
        
        os.makedirs(self.BASE_DIR, exist_ok=True)
        os.makedirs(self.MODEL_BASE_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        

    

config = Config()
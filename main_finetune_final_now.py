import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime
import wandb
import gc

from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
from config import Config
from data_utils import preprocess_data_meld
from visualization import visualize_results
from train_utils import log_metrics, evaluate_model
import torch

from models import get_model, print_model_info, unfreeze_layers, EmotionRecognitionModel_v2, EmotionRecognitionWithWav2Vec 

gc.collect()
torch.cuda.empty_cache()

def save_model(model, path):
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(path)
    else:
        torch.save(model.state_dict(), os.path.join(path, 'model.pt'))

def load_model(model, path):
    if hasattr(model, 'from_pretrained'):
        return model.from_pretrained(path)
    else:
        model.load_state_dict(torch.load(os.path.join(path, 'model.pt')))
        return model
    
def get_logits_from_output(outputs):
    if isinstance(outputs, dict):
        return outputs.get('logits', outputs.get('last_hidden_state'))
    elif isinstance(outputs, torch.Tensor):
        return outputs  # 이미 로짓 텐서인 경우
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    elif hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    else:
        raise ValueError("Unexpected output format from the model")
    
def save_log(log_data, filename):
    with open(filename, 'w') as f:
        json.dump(log_data, f, indent=4)

def load_and_preprocess_audio(file_path, max_length=80000): 
    waveform, sample_rate = torchaudio.load(str(file_path))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
  
    if waveform.shape[0] > max_length:
        waveform = waveform[:max_length]
    else:
        padding = torch.zeros(max_length - waveform.shape[0])
        waveform = torch.cat([waveform, padding])
    
    return waveform

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = [str(path) for path in file_paths]
        self.labels = labels.tolist() if isinstance(labels, np.ndarray) else labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio = load_and_preprocess_audio(self.file_paths[idx])
        return {"audio": audio, "label": self.labels[idx]}
  
def collate_fn(batch):
    if isinstance(batch[0], dict):
        # 배치 아이템이 딕셔너리 형태일 경우
        audio = [item['audio'] for item in batch]
        labels = [item['label'] for item in batch]
    else:
        # 배치 아이템이 튜플 형태일 경우
        audio, labels = zip(*batch)
    
    # audio 텐서로 변환 및 패딩
    audio_tensors = [torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a for a in audio]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    
    # 레이블을 텐서로 변환
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"audio": audio_padded, "label": labels_tensor}
def train(model, train_dataloader, val_dataloader, config):
    device = config.device
    best_val_f1 = 0
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }
    optimizer_grouped_parameters = [
    {'params': model.wav2vec2.parameters(), 'lr': 1e-5, 'weight_decay':config.weight_decay},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay':config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        config.global_epoch+=1
        if config.global_epoch == 5:
            unfreeze_layers(model, 6)  # 5번째 에폭 후 6개 레이어 동결 해제
        elif config.global_epoch == 10:
            unfreeze_layers(model, 9)  # 10번째 에폭 후 9개 레이어 동결 해제
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_dataloader:
            audio_input = batch['audio'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio_input) # type diff/ SeqClassifier
            logits = get_logits_from_output(outputs)#outputs.logits#['logits']
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Validation
        train_metrics =  evaluate_model(config, model, train_dataloader, criterion, device)
        val_metrics =  evaluate_model(config, model, val_dataloader, criterion, device)
            
        # Update history
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            history['train'][metric].append(val_metrics[i])
            history['val'][metric].append(val_metrics[i])
            print(f"Epoch {config.global_epoch}/{config.NUM_EPOCHS}:")
    
        print(f"Train - Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}, F1: {train_metrics[4]:.4f}")
        print(f"Val - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}, F1: {val_metrics[4]:.4f}")
        
        ######
        #Log metrics chk
        log_metrics('train', train_metrics, config.global_epoch)
        log_metrics('val', val_metrics[:5], config.global_epoch)  # val_metrics might have 7 values, we only need first 5
        
        if config.global_epoch % config.N_STEP_FIG ==0: # visualization for val data
            try:
                visualize_results(config, model, val_dataloader, device, history, 'val')
            except Exception as e:
                print(f"Error during visualization: {e}")         
        
        # 최고 성능 모델 저장
        if train_metrics[4] > best_val_f1:
            best_val_f1 = train_metrics[4]
            try:
                save_model(model, config.path_best)
                #model.save_pretrained(config.path_best)
            except:
                print('err.')
    
    return model, history
# class Wav2VecFeatureExtractor(torch.nn.Module):
#     def __init__(self, model_name="facebook/wav2vec2-base"):
#         super().__init__()
#         self.model = Wav2Vec2Model.from_pretrained(model_name)
        
#     def forward(self, input_values):
#         outputs = self.model(input_values, output_hidden_states=True)
#         # penultimate layer (-2)의 hidden state를 반환
#         return outputs.hidden_states[-2]
# class Wav2Vec2ClassifierModel(nn.Module):
#     def __init__(self, config, num_labels, dropout=0.4):
#         super().__init__()
#         self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.path_best)
#         self.classifier = nn.Sequential(
#             nn.Linear(768, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_labels)
#         )
    
#     def forward(self, input_values):
#         outputs = self.wav2vec2(input_values)
#         hidden_states = outputs.last_hidden_state
#         pooled_output = torch.mean(hidden_states, dim=1)
#         logits = self.classifier(pooled_output)
#         return logits  # 직접 logits를 반환



### 
config=Config()

config.N_STEP_FIG=1

num_epochs = 60
config.NUM_EPOCHS=num_epochs
#lr=1e-4
n_batch = 8 # 74% GPU. 8 is danger high n_batch -> small batch size -> low gpu?
config.BATCH_SIZE=n_batch
n_labels = len(config.LABELS_EMO_MELD)

#wav2vec_path = ".models/wav2vec2_finetuned"  # 파인튜닝된 wav2vec2 모델 경로
wav2vec_path="facebook/wav2vec2-base"

data_dir=os.path.join(config.DATA_DIR, 'MELD', 'train_audio')
label_dir=os.path.join(config.DATA_DIR, 'MELD_train_sampled.csv')
# load class label info
label_info_df= pd.read_csv(label_dir)

print(f'Data location: {data_dir}\nlabel info: {label_dir}')

# Data prep
file_paths, labels = preprocess_data_meld(data_dir, label_info_df)

dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
labels=[dict_label[val] for val in labels]
config.LABELS_EMOTION =config.LABELS_EMO_MELD

dataset = AudioDataset(file_paths, labels)

# 데이터셋 분할
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=n_batch, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
config.device = device

###### I.
# Fine-tuning 및 성능 기록

#model = Wav2VecFeatureExtractor()

# model= EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_dataloader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE, activation=config.ACTIVATION, use_wav2vec=True)

# wandb log
config.MODEL= 'wav2vec_finetuned'
config.DATA_NAME='MELD'
config.WANDB_PROJECT=config.MODEL+'_'+config.DATA_NAME

path_best = f"{os.path.join(config.MODEL_PRE_BASE_DIR, config.WANDB_PROJECT)+'_best'}"
os.makedirs(path_best, exist_ok=True)
print(path_best)
config.update_path()

model = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_path, num_labels=n_labels, output_hidden_states=True)
model.to(device)
for param in model.parameters():
    param.requires_grad = False
n_unfreeze=3
unfreeze_layers(model, n_unfreeze)
config.lr =1e-4

print(config.MODEL_DIR)

config_wandb = {'lr': config.lr,
                'n_batch': n_batch, 
                'model_path': config.MODEL_DIR
                }
id_wandb = wandb.util.generate_id()
print(f'Wandb id generated: {id_wandb}')
config.id_wandb = id_wandb
wandb.init(id=id_wandb, project=config.WANDB_PROJECT)#, config=config.CONFIG_DEFAULTS)

model, log_data = train(model, train_dataloader, val_dataloader, config)
# 최종 모델 저장

try:
    save_model(model, config.MODEL_DIR)
    #model.save_pretrained(os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))
except:
    print('save err')
    # torch.save(model.state_dict(), os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))

# 학습 로그 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_log(log_data, os.path.join(config.MODEL_PRE_BASE_DIR, f"training_log_{timestamp}.json"))

gc.collect()
torch.cuda.empty_cache()
wandb.finish()

# #### II. 
# config.NUM_EPOCHS = 60
# config.model_name = 'wav2vec_II'
# config.lr = 5e-5
# config.DROPOUT_RATE = 0.4

# # for temp
# config.path_best = os.path.join(config.MODEL_BASE_DIR, 'wav2vec2_finetuned')
# print(config.path_best)
# new_model = Wav2Vec2ClassifierModel(config, num_labels=n_labels, dropout=config.DROPOUT_RATE)
# new_model.to(device)

# config.WANDB_PROJECT = 'wav2vec_II_classifier_0'
# config.MODEL_DIR = os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)
# os.makedirs(config.MODEL_DIR, exist_ok=True)

# id_wandb = wandb.util.generate_id()
# print(f'Wandb id generated: {id_wandb}')
# config.id_wandb = id_wandb
# config_wandb = {'lr': config.lr,
#                 'dropout': config.DROPOUT_RATE,
#                 'n_batch': config.BATCH_SIZE}
# try:
#     wandb.init(id=id_wandb, config=config_wandb, project=config.WANDB_PROJECT)
# except Exception as e:
#     print(f"Failed to initialize wandb: {e}")

# new_model, log_data = train(new_model, train_dataloader, val_dataloader, config)

# path_new=os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)
# os.makedirs(path_new, exist_ok=True)

# try:
#     save_model(new_model,path_new )
# except:
#     print('save err')
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# save_log(log_data, f"training_log_{timestamp}.json")

# gc.collect()
# torch.cuda.empty_cache()
# wandb.finish()

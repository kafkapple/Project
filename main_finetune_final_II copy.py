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
from train_utils import log_metrics
import torch

gc.collect()
torch.cuda.empty_cache()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            audio_input = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(audio_input)
            hidden_states = outputs['last_hidden_state']
            pooled_output = torch.mean(hidden_states, dim=1)
            logits = model.classifier(pooled_output)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

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
    audio = [item['audio'] for item in batch]
    audio = pad_sequence(audio, batch_first=True, padding_value=0.0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {"audio": audio, "label": labels}

def train(model, train_dataloader, val_dataloader, config):
    device = config.device
    best_val_f1 = 0
    log_data = {
        'train': {'loss': [], 'accuracy': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'f1': []}
    }
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    path_best = f"{os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)+'_best'}"
    os.makedirs(path_best, exist_ok=True)
    config.path_best = path_best
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_dataloader:
            audio_input = batch['audio'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio_input)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_dataloader)
        train_accuracy = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Validation
        val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader, criterion, device)
        
        # 로그 데이터 저장
        log_data['train']['loss'].append(avg_train_loss)
        log_data['train']['accuracy'].append(train_accuracy)
        log_data['train']['f1'].append(train_f1)
        log_data['val']['loss'].append(val_loss)
        log_data['val']['accuracy'].append(val_accuracy)
        log_data['val']['f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        config.global_epoch = epoch + 1
        visualize_results(config, model, val_dataloader, device, log_data, 'val')
        log_metrics('train', log_data['train'], config.global_epoch)
        log_metrics('val', log_data['val'], config.global_epoch)
        
        config.global_epoch=epoch+1
        visualize_results(config, model, val_dataloader, device, log_data, 'val')
        log_metrics('train', log_data['train'], config.global_epoch)
        log_metrics('val', log_data['val'], config.global_epoch)
        
        # 최고 성능 모델 저장
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            try:
                model.save_pretrained(config.path_best)
            except:
                torch.save(model.state_dict(), config.path_best)
        # 주기적으로 모델 저장 (예: 5 에폭마다)
        if (epoch + 1) % config.N_STEP_FIG == 0:
            path_epoch = f"{os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)}_epoch_{epoch+1}"
            os.makedirs(path_epoch, exist_ok=True)
            try:
                model.save_pretrained(path_epoch)
            except:
                torch.save(model.state_dict(), path_epoch)
            
    return model, log_data

class Wav2Vec2ClassifierModel(nn.Module):
    def __init__(self, config, num_labels, dropout=0.4):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.path_best)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )
        

    
    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        return {'last_hidden_state': hidden_states, 'logits': self.classifier(pooled_output)}


### 
config=Config()
num_epochs = 30
config.NUM_EPOCHS=num_epochs
#lr=1e-4
n_batch = 4 # 74% GPU. 8 is danger high n_batch -> small batch size -> low gpu?
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
config.model_name= 'wav2vec_I'
model = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_path, num_labels=n_labels)
model.to(device)
config.lr =1e-4
# wandb log
config.WANDB_PROJECT='wav2vec_I_fine_tune'
config.MODEL_DIR = os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)
os.makedirs(config.MODEL_DIR, exist_ok=True)

config_wandb = {'lr': config.lr,
                'n_batch': n_batch
                }
id_wandb = wandb.util.generate_id()
print(f'Wandb id generated: {id_wandb}')
config.id_wandb = id_wandb
wandb.init(id=id_wandb, project=config.WANDB_PROJECT)#, config=config.CONFIG_DEFAULTS)

model, log_data = train(model, train_dataloader, val_dataloader, config)
# 최종 모델 저장
model.save_pretrained(os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))

# 학습 로그 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_log(log_data, f"training_log_{timestamp}.json")

gc.collect()
torch.cuda.empty_cache()
wandb.finish()


config.NUM_EPOCHS = 60
config.model_name = 'wav2vec_II'
config.lr = 5e-5
config.DROPOUT_RATE = 0.4
config.path_best = os.path.join(config.MODEL_BASE_DIR, 'wav2vec2_finetuned')
print(config.path_best)
new_model = Wav2Vec2ClassifierModel(config, num_labels=n_labels, dropout=config.DROPOUT_RATE)
new_model.to(device)

config.WANDB_PROJECT = 'wav2vec_II_classifier_0'
config.MODEL_DIR = os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)
os.makedirs(config.MODEL_DIR, exist_ok=True)

id_wandb = wandb.util.generate_id()
print(f'Wandb id generated: {id_wandb}')
config.id_wandb = id_wandb
config_wandb = {'lr': config.lr,
                'dropout': config.DROPOUT_RATE,
                'n_batch': config.BATCH_SIZE}
try:
    wandb.init(id=id_wandb, config=config_wandb, project=config.WANDB_PROJECT)
except Exception as e:
    print(f"Failed to initialize wandb: {e}")

new_model, log_data = train(new_model, train_dataloader, val_dataloader, config)
new_model.save_pretrained(os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_log(log_data, f"training_log_{timestamp}.json")

gc.collect()
torch.cuda.empty_cache()
wandb.finish()

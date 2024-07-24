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

from models import get_model, EmotionRecognitionModel_v2, EmotionRecognitionWithWav2Vec 

gc.collect()
torch.cuda.empty_cache()

def print_model_info(model):
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Structure:")
    print(model)
    print("\nLayer-wise details:")
    
    for name, module in model.named_modules():
        if not list(module.children()):  # 자식 모듈이 없는 경우 (즉, 기본 레이어인 경우)
            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")
            params = sum(p.numel() for p in module.parameters())
            print(f"Parameters: {params:,}")

    print(f"\nTotal layers: {len(list(model.modules()))}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
def unfreeze_layers(model, num_layers):
    for param in model.parameters():
        param.requires_grad = False
    
    for i, layer in enumerate(reversed(list(model.wav2vec2.encoder.layers))):
        if i < num_layers:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            break
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
            #print(type(outputs), outputs)
            logits = get_logits_from_output(outputs)  # 변경된 부분
            
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
  
# def collate_fn(batch):
#     audio = [item['audio'] for item in batch]
#     audio = pad_sequence(audio, batch_first=True, padding_value=0.0)
#     labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
#     return {"audio": audio, "label": labels}
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
    log_data = {
        'train': {'loss': [], 'accuracy': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'f1': []}, 'epoch':[]
    }
    
    
    optimizer_grouped_parameters = [
    {'params': model.wav2vec2.parameters(), 'lr': 1e-5, 'weight_decay':config.weight_decay},
    {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay':config.weight_decay}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    
    path_best = f"{os.path.join(config.MODEL_BASE_DIR,'finetuned', config.WANDB_PROJECT)+'_best'}"
    os.makedirs(path_best, exist_ok=True)
    config.path_best = path_best
    
    for epoch in tqdm(range(config.NUM_EPOCHS)):
        if epoch == 5:
            unfreeze_layers(model, 6)  # 5번째 에폭 후 6개 레이어 동결 해제
        elif epoch == 10:
            unfreeze_layers(model, 9)  # 10번째 에폭 후 9개 레이어 동결 해제
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in train_dataloader:
            audio_input = batch['audio'].to(device)
            labels = batch['label'].to(device)

            outputs = model(audio_input)
            logits = get_logits_from_output(outputs)#outputs.logits#['logits']
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
        log_data['epoch'].append(config.global_epoch)
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
                save_model(model, config.path_best)
                #model.save_pretrained(config.path_best)
            except:
                print('err.')
                #torch.save(model.state_dict(), config.path_best)
        # 주기적으로 모델 저장 (예: 5 에폭마다)
        if (epoch + 1) % config.N_STEP_FIG == 0:
            path_epoch = f"{os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)}_epoch_{epoch+1}"
            os.makedirs(path_epoch, exist_ok=True)
            try:
                save_model(model, path_epoch)
                #model.save_pretrained(path_epoch)
            except:
                print('err.')
               # torch.save(model.state_dict(), path_epoch)
            
    return model, log_data
class Wav2VecFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        
    def forward(self, input_values):
        outputs = self.model(input_values, output_hidden_states=True)
        # penultimate layer (-2)의 hidden state를 반환
        return outputs.hidden_states[-2]
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



num_epochs = 60
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
config.model_name= 'wav2vec_I_0'
model = Wav2VecFeatureExtractor()

# model= EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_dataloader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE, activation=config.ACTIVATION, use_wav2vec=True)

#model = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_path, num_labels=n_labels)
model.to(device)
for param in model.parameters():
    param.requires_grad = False
n_unfreeze=3
unfreeze_layers(model, n_unfreeze)
config.lr =1e-4
print_model_info(model)

# wandb log
config.WANDB_PROJECT='wav2vec_I_fine_tune'
config.MODEL_DIR = os.path.join(config.MODEL_BASE_DIR, 'finetuned',config.WANDB_PROJECT)
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
new_path=os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT)
os.makedirs(new_path, exist_ok=True)
try:
    save_model(model, new_path)
    #model.save_pretrained(os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))
except:
    print('save err')
    # torch.save(model.state_dict(), os.path.join(config.MODEL_BASE_DIR, config.WANDB_PROJECT))

# 학습 로그 저장
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_log(log_data, f"training_log_{timestamp}.json")

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

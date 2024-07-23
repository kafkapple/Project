import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio.transforms as T
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torchaudio
import numpy as np
from tqdm import tqdm
import os
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data import DataLoader

from config import Config
from data_utils import preprocess_data_meld
import numpy as np
import torch
import torchaudio

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # 패딩 적용
    input_values = pad_sequence(input_values, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    labels = torch.stack(labels)
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }
def load_model(model_path, device, config):
    model = Wav2Vec2Classifier(Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base'), config.num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, optimizer, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}")
        return epoch
    else:
        print("No checkpoint found. Starting from scratch.")
        return 0

# 1. 모델 정의
class Wav2Vec2Classifier(nn.Module):
    def __init__(self, wav2vec_model, num_classes):
        super(Wav2Vec2Classifier, self).__init__()
        self.wav2vec = wav2vec_model
        self.classifier = nn.Linear(self.wav2vec.config.hidden_size, num_classes)
        
    def forward(self, input_values, attention_mask):
        if input_values.dim() == 4:
          input_values = input_values.squeeze(1)  # [batch_size, 1, 1, sequence_length] -> [batch_size, 1, sequence_length]
      
        outputs = self.wav2vec(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled_output)
        return logits

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, labels, feature_extractor, max_length=16000*10):  
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.resampler = T.Resample(orig_freq=44100, new_freq=16000)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 리샘플링 적용
        if sample_rate != 16000:
            waveform = self.resampler(waveform)
        
        waveform = waveform.squeeze()

        # 오디오 길이 제한
        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            # 패딩
            waveform = F.pad(waveform, (0, self.max_length - waveform.size(0)))
        
        # feature_extractor에 패딩 작업 위임
        # inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        # input_values = inputs.input_values.squeeze(0)
        
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=self.max_length)
        input_values = inputs.input_values.squeeze()  # squeeze() 대신 squeeze(0) 사용
        print(input_values.shape)
        # attention_mask가 없다면 직접 생성
        if 'attention_mask' in inputs:
            attention_mask = inputs.attention_mask.squeeze()  # squeeze() 대신 squeeze(0) 사용
        else:
            attention_mask = torch.ones_like(input_values)

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
# 3. 학습 함수
def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, config):
    criterion = nn.CrossEntropyLoss()
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  # 'label'에서 'labels'로 변경


            optimizer.zero_grad()
            outputs = model(input_values, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss:.4f}")
        
        val_accuracy, val_f1 = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}, F1 Score: {val_f1:.4f}")
        save_checkpoint(model, optimizer, epoch + 1, config.CKPT_SAVE_PATH)

# 4. 평가 함수
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_values, attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

# 5. 메인 파이프라인
def main():
    # config
    config=Config()
    data_dir=os.path.join(config.DATA_DIR, 'MELD.Raw', 'train_audio')
    label_dir=os.path.join(config.DATA_DIR, 'MELD_train_sampled.csv')
    # load class label info
    label_info_df= pd.read_csv(label_dir)
    
    config.CKPT_SAVE_PATH = os.path.join(config.MODEL_BASE_DIR,'pretrained', 'trained_model.pth' )
    os.makedirs(config.CKPT_SAVE_PATH, exist_ok=True)

    print(f'Data location: {data_dir}\nlabel info: {label_dir}')
    
    pretrained_model_name = "./wav2vec2_finetuned"  # 파인튜닝된 wav2vec2 모델 경로
    # Model 
 
    pretrained_model_name = 'facebook/wav2vec2-base'
    num_classes = len(config.LABELS_EMO_MELD) # 분류할 클래스 수
    config.num_classes=num_classes
    
    batch_size = 4 # 74% GPU. 8 is danger high batch_size -> small batch size -> low gpu?
    num_epochs = 5
    learning_rate = 1e-4
    n_sec = 5
    max_length = 16000 * n_sec  
    

    # 특징 추출기와 프리트레인된 모델 로드
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name)
    
    # Data prep
    print('Data prep...')
    file_paths, labels = preprocess_data_meld(data_dir, label_info_df)#preprocess_data
    dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
    labels=[dict_label[val] for val in labels]
    config.LABELS_EMOTION =config.LABELS_EMO_MELD
    
    # 데이터 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(file_paths, labels, test_size=0.2)
    print('Splitted')
    # 데이터셋 준비
    train_dataset = AudioDataset(train_paths, train_labels, feature_extractor, max_length)
    val_dataset = AudioDataset(val_paths, val_labels, feature_extractor, max_length)

    # 데이터로더 생성
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('pretrained model loaded')
    wav2vec_model = Wav2Vec2Model.from_pretrained(pretrained_model_name)
    
    #model = Wav2Vec2ForSequenceClassification.from_pretrained(wav2vec_path, num_labels=num_classes)

    #model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=n_labels)

    model = Wav2Vec2Classifier(wav2vec_model, num_classes).to(device)

    # 옵티마이저와 스케줄러 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)

    # 학습 실행
    train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, num_epochs, config)

    # 최종 평가
    final_accuracy, final_f1 = evaluate(model, val_dataloader, device)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}, F1 Score: {final_f1:.4f}")

if __name__ == "__main__":
    main()
        #loaded_model = load_model("wav2vec_classifier_final.pth", device)

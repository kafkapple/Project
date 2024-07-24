from transformers import Wav2Vec2ForSequenceClassification#, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
# custom func
from config import Config
from data_utils import preprocess_data_meld

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

import gc


gc.collect()
torch.cuda.empty_cache()


def load_and_preprocess_audio(file_path, max_length=80000): # max num of audio is fixed (5 sec)
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
# config
config=Config()
print(config)
data_dir=os.path.join(config.DATA_DIR, 'MELD', 'train_audio')
label_dir=os.path.join(config.DATA_DIR, 'MELD_train_sampled.csv')
# load class label info
label_info_df= pd.read_csv(label_dir)


print(f'Data location: {data_dir}\nlabel info: {label_dir}')

# Data prep
file_paths, labels = preprocess_data_meld(data_dir, label_info_df)#preprocess_data(data_dir)
#print(type(file_paths), type(file_paths[0]))

dict_label = {v: k for k, v in config.LABELS_EMO_MELD.items()} 
labels=[dict_label[val] for val in labels]
config.LABELS_EMOTION =config.LABELS_EMO_MELD

dataset = AudioDataset(file_paths, labels)
# Test
# first_item = dataset[0]
# print("First item audio type:", type(first_item['audio']))
# print("First item audio shape:", first_item['audio'].shape)
# print("First item label:", first_item['label'])

# test_batch = [dataset[i] for i in range(4)]  
# collated = collate_fn(test_batch)
# print("Collated audio shape:", collated['audio'].shape)
# print("Collated labels:", collated['label'])

n_batch = 4 # 74% GPU. 8 is danger high n_batch -> small batch size -> low gpu?
n_labels = len(config.LABELS_EMO_MELD)
dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True, collate_fn=collate_fn)

# Model 
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=n_labels)
# Fine-tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

num_epochs = 3
for epoch in tqdm(range(num_epochs)):
  
    # train_loss, train_acc, train_f1 = train_epoch(model, dataloader, optimizer, criterion, device)
    model.train()
    total_loss = 0
    for batch in dataloader:
        audio_input = batch['audio'].to(device)
        labels = batch['label'].to(device)

        outputs = model(audio_input)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss+=loss.item()
        #val_loss, val_acc, val_f1 = validate(model, val_dataloader, criterion, device)
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}: {avg_loss}")
    #print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
    # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    #print("-" * 50)

model.save_pretrained("./models/fine_tuning/wav2vec2")
gc.collect()
torch.cuda.empty_cache()
import os
import numpy as np
import torch
import torchaudio
import requests
import zipfile
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils import resample
from collections import Counter

from tqdm import tqdm

from config import Config
import pandas as pd
import nltk
import spacy
import string
import tarfile

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# preprocess the data (common across all models)
from sentence_transformers import SentenceTransformer

# # word2vec
# #from gensim.models import Word2Vec
# # url = "https://raw.githubusercontent.com/ataislucky/Data-Science/main/dataset/emotion_train.txt"
        
from sklearn.preprocessing import StandardScaler
import moviepy.editor as mp

import os
from tqdm import tqdm
import moviepy.editor as mp

def get_logits_from_output(outputs):
    if isinstance(outputs, dict):
        return outputs.get('logits', outputs.get('last_hidden_state', outputs))
    elif isinstance(outputs, torch.Tensor):
        return outputs  # 이미 로짓 텐서인 경우
    elif hasattr(outputs, 'logits'):
        return outputs.logits
    elif hasattr(outputs, 'last_hidden_state'):
        return outputs.last_hidden_state
    else:
        return outputs  # 예상치 못한 형식이지만 그대로 반환

def prep_audio(config, text_train_df, destination_base_path, TARGET):  # by Lek Hong
    
    os.makedirs(destination_base_path, exist_ok=True)
    
    if TARGET == 'train':
        TARGET_SPLIT = 'train_splits'
    elif TARGET == 'test':
        TARGET_SPLIT = 'output_repeated_splits_test'
    else:
        print('No target specified.')
        return

    for dialogue_id, utterance_id in tqdm(zip(text_train_df["Dialogue_ID"], text_train_df["Utterance_ID"])):
        # Source and destination paths
        source_path = os.path.join(config.DATA_DIR, 'MELD.Raw', TARGET, TARGET_SPLIT, f'dia{dialogue_id}_utt{utterance_id}.mp4')
        destination_path = os.path.join(destination_base_path, f'dia{dialogue_id}_utt{utterance_id}.wav')
        
        # Check if the destination audio file already exists
        if os.path.exists(destination_path):
            print(f"File {destination_path} already exists. Skipping...")
            continue

        try:
            # Load the video file
            video = mp.VideoFileClip(source_path)

            # Extract the audio
            audio = video.audio

            if audio is None:
                print(f"Audio extraction failed for {source_path}. Skipping...")
                video.close()
                continue

            # Write the audio to the destination path
            audio.write_audiofile(destination_path)

            # Close the video and audio objects to release resources
            audio.close()
            video.close()

        except Exception as e:
            print(f"Error processing {source_path}: {e}")
            continue

        audio.write_audiofile(destination_path)

def extract_features_and_labels(dataloader):
    all_features = []
    all_labels = []
    for features, labels in tqdm(dataloader, desc="Extracting features"):
      # 만약 features가 3D (batch, sequence_length, feature_dim)라면 2D로 변환
      if features.dim() == 3:
          features = features.view(features.size(0), -1)
      all_features.append(features.cpu().numpy())
      all_labels.append(labels.cpu().numpy())
      return np.vstack(all_features), np.concatenate(all_labels)

def prep_data_for_benchmark(data_loader):
    X, y= extract_features_and_labels(data_loader)
    print(f"Shape of X_train: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def convert_to_int_keys(dictionary):
    """
    :param dictionary: 
    :return: 
    """
 
    if all(isinstance(k, int) for k in dictionary.keys()):
        print("Keys are already integers. Returning original dictionary.")
        return dictionary
    

    if all(isinstance(k, str) for k in dictionary.keys()) and all(isinstance(v, int) for v in dictionary.values()):
        print("Swapping keys and values.")
        return {v: k for k, v in dictionary.items()}
    

    if not any(isinstance(k, int) or isinstance(v, int) for k, v in dictionary.items()):
        print("No integer type found in keys or values. Returning original dictionary.")
        return dictionary
    

    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(k, int):
            new_dict[k] = v
        elif isinstance(v, int):
            new_dict[v] = k
        else:
            new_dict[k] = v  # 정수가 아닌 경우 그대로 유지
    
    print("Converted dictionary to have integer keys where possible.")
    return new_dict


class TextDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features = self.data[idx]
        label = self.labels[idx]
        return features, label
    
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
#wav2vec2_model.gradient_checkpointing_enable()

def download_dataset(config):
    
    url = config.dataset[config.DATA_NAME]#"https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    if config.DATA_NAME == 'RAVDESS':
        dataset_ext = 'zip'
        extracted_path = os.path.join(config.DATA_DIR, config.DATA_NAME)
                                      
    elif config.DATA_NAME == 'MELD':
        dataset_ext = 'tar.gz'
        extracted_path = os.path.join(config.DATA_DIR, f"{config.DATA_NAME}.Raw")
    dataset_path= extracted_path + '.'+dataset_ext
    print('Extraction path: ', extracted_path, dataset_path)
    config.extracted_path=extracted_path

    print(f'Dataset: {config.DATA_NAME}')
    if not os.path.exists(dataset_path):
        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(dataset_path, 'wb') as f:
                f.write(response.content)
            print("Download complete.")
        else:
            print("Failed to download the dataset.")
            return config.DATA_DIR, False
    else:
        print("Dataset already exists. Skipping download.\n")
    TARGET=config.TARGET
    print(extracted_path)
    if not os.path.exists(extracted_path):
        print('Data extraction starts.')
        if dataset_ext == 'zip':
            try:
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_path)
                print("Extracted dataset.")
            except Exception as e:
                print(f"Extraction failed: {e}")
                return config.DATA_DIR, False
        elif dataset_ext =='tar.gz':
            try:
                with tarfile.open(dataset_path, 'r:gz') as tar:  # gzip 압축된 tar 파일
                    tar.extractall(path=config.DATA_DIR)
                print("Extracted main dataset.")
                
            except Exception as e:
                print(f"Extraction failed - main: {e}")
                return config.DATA_DIR, False
            
    else:
        print("Dataset already extracted.\n")
        
    if config.DATA_NAME=='MELD' and not os.path.exists(os.path.join(extracted_path, f'{TARGET}')):
        print('######### MELD')
        try:
            path_tar2=os.path.join(extracted_path, f'{TARGET}.tar.gz')
            print(path_tar2)
                                
            with tarfile.open(path_tar2, 'r:gz') as tar:  # gzip 압축된 tar 파일
                path_tar3=os.path.join(extracted_path, f'{TARGET}')
                print(path_tar3)
                tar.extractall(path=path_tar3)
            print("Extracted sub dataset.")
            
        except Exception as e:
            print(f"Extraction failed - sub: {e}")
            return config.DATA_DIR, False
    return extracted_path, True

def preprocess_data_meld(data_dir, text_train_df):
    data = []
    labels = []
    count=0
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                data.append(file_path)

                dialogue_id = int(file.split("_")[0][3:])
                utterance_id = int(file.split("_")[1].split(".")[0][3:])
                try:
                    label = text_train_df.loc[(text_train_df["Dialogue_ID"]==dialogue_id) & (text_train_df["Utterance_ID"]==utterance_id), "Emotion"].item()
                    labels.append(label)
                except:
                    print('No file or err')
                    count+=1
    print('Total err: ', count)

    if len(data) == 0:
        raise ValueError("No valid .wav files found in the dataset.")
    return np.array(data), np.array(labels)

def preprocess_data(data_dir):
    data = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                label = int(file.split('-')[2]) - 1
                data.append(file_path)
                labels.append(label)
    if len(data) == 0:
        raise ValueError("No valid .wav files found in the dataset.")
    return np.array(data), np.array(labels)

def extract_features(waveform, sample_rate):
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = wav2vec2_model(**inputs)
    wav2vec2_features = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()
    return wav2vec2_features.reshape(1, -1)  # reshape (1, input_size) 

class AudioDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        features = extract_features(waveform, sample_rate)
        return features, label
def collate_fn(batch): 
    if isinstance(batch[0], dict): # if dict
    
        audio = [item['audio'] for item in batch]
        labels = [item['label'] for item in batch]
    else: # if tuple
   
        audio, labels = zip(*batch)
    
    # audio tensor conver and padding to match length - max length in batch 
    audio_tensors = [torch.tensor(a, dtype=torch.float32) if not isinstance(a, torch.Tensor) else a for a in audio]
    audio_padded = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)
    
    # 레이블을 텐서로 변환
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    return {"audio": audio_padded, "label": labels_tensor}

def prepare_dataloaders(data, labels, config, combine_indices=None, balance=False):
    if combine_indices:
        labels = combine_labels(labels, combine_indices)
    
    if balance:
        data, labels = balance_classes(data, labels)
    
    full_dataset = AudioDataset(data, labels)
    
    train_size = int(config.RATIO_TRAIN * len(full_dataset))
    val_size = int(config.RATIO_TEST * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config.SEED))
    
    print(f"\nTrain/Val/Test set splitted with batch size {config.BATCH_SIZE}: {len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)}\n")
    
    # print('Calculating label distributions...')
    if config.VISUALIZE:
        train_labels = [labels[idx] for idx in train_dataset.indices]
        val_labels = [labels[idx] for idx in val_dataset.indices]
        test_labels = [labels[idx] for idx in test_dataset.indices]
        print_label_distribution(train_labels, "Train")
        print_label_distribution(val_labels, "Validation")
        print_label_distribution(test_labels, "Test")
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

def print_label_distribution(labels, dataset_name="Dataset"):
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"\n{dataset_name} label distribution:")
    for label, count in counter.items():
        print(f"Label {label}: {count} samples ({count / total:.2%})")

def combine_labels(config, class_info, labels, combine_indices):
    keys_to_merge = sorted(set(combine_indices))
    new_dict = {}
    for key in class_info:
        if key in keys_to_merge:
            min_key = min(k for k in keys_to_merge if k in class_info)
            merged_value = '_'.join(class_info[k] for k in keys_to_merge if k in class_info)
            new_dict[min_key] = merged_value
        else:
            new_dict[key] = class_info[key]

    key_mapping = {k: min(keys_to_merge) for k in keys_to_merge}
    new_labels = [key_mapping.get(x, x) for x in labels]

    count_dict = {}
    for num in new_labels:
        if num in new_dict:
            count_dict[num] = count_dict.get(num, 0) + 1

    for key, count in count_dict.items():
        print(f"{key}. {new_dict[key]}: {count}")
        
    config.LABELS_EMOTION = new_dict

    return new_dict, new_labels


def balance_classes(data, labels):
    unique_labels = np.unique(labels)
    max_samples = max(Counter(labels).values())
    
    new_data = []
    new_labels = []
    
    for label in unique_labels:
        label_data = data[labels == label]
        label_labels = labels[labels == label]
        
        if len(label_data) < max_samples:
            label_data, label_labels = resample(label_data, label_labels, replace=True, n_samples=max_samples, random_state=42)
        
        new_data.append(label_data)
        new_labels.append(label_labels)
    
    return np.concatenate(new_data), np.concatenate(new_labels)

def load_data(config):#, dataset):
    data_dir, status = download_dataset(config)#, dataset)
    if not status:
        raise Exception("Failed to download or extract the dataset.")
    return data_dir

#### Text
def preprocess_text(text):
      # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word.lower() not in stop_words]
    # Rejoin tokens to create the cleaned sentence
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# def data_prep_text(config):
#     st_encoder = SentenceTransformer('all-MiniLM-L12-v2')
#       # Download necessary NLTK data
#     nltk.download('stopwords')
#     nltk.download('wordnet')
#     # # Load spaCy model
#     # nlp = spacy.load('en_core_web_sm')

#     # Initialize stop words and lemmatizer
#     stop_words = set(stopwords.words('english'))
#     lemmatizer = WordNetLemmatizer()
    
#     data_1 = pd.read_csv(os.path.join(config.DATA_DIR,"val.txt"), sep=";")
#     data_1.columns = ["Text", "Emotions"]
#     data_3 = pd.read_csv(os.path.join(config.DATA_DIR,"validation.csv")).rename(columns={"text":"Text", "label":"Emotions"})
#     data_3["Emotions"] = data_3["Emotions"].map({0:"sadness", 1:"joy", 2:"love", 3:"anger", 4:"fear", 5:"surprise"})

#     # Train Word2Vec model
#     data_1_3 = pd.concat([data_1, data_3])
#     sentences = [sentence.split() for sentence in data_1_3['Text']]

#     VECTOR_SIZE = 100
#     MIN_COUNT = 5
#     WINDOW = 3
#     SG = 1
#     w2v_model = Word2Vec(sentences, vector_size=VECTOR_SIZE, min_count=MIN_COUNT, window=WINDOW, sg=SG)

#     # encode the data
#     data_1_3['Cleaned_Text'] = data_1_3['Text'].apply(preprocess_text).dropna()
#     data_1_3["hf_embed"] = data_1_3['Cleaned_Text'].apply(lambda x: st_encoder.encode(x))

    # Obtain word embeddings for data_1.Text and train a svm model on it with class being data_1.Emotion and measure accuracy
    # Apply preprocessing to the text data
    data_1_3['Cleaned_Text'] = data_1_3['Text'].apply(preprocess_text).dropna()

#     # Get word embeddings for the cleaned text
#     X = data_1_3['Cleaned_Text'].apply(lambda sent: w2v_model.wv.get_mean_vector([word for word in sent.split()]))
#     X = np.stack(X.values)

#     # Continue with mapping emotions to numerical labels, and splitting the data
#     y = data_1_3['Emotions'].map(config.LABELS_EMOTION_TEXT).values
#     #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     # Get word embeddings for data.Text
#     data = data_1_3.copy()
#     X = data['hf_embed']
#     X = np.stack(X.values)
#     # Map emotions to numerical labels
#     y = data['Emotions'].map(config.LABELS_EMOTION_TEXT).values
#     print(f'Data size: {X.shape, y.shape}\nPreparing Text dataloader...')
#     # Split data into train and test sets
    
#     # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
#     # X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)
    
#     train_loader, val_loader, test_loader=prepare_dataloaders(X, y, config)
#     return train_loader, val_loader, test_loader#_train, X_val, X_test, y_train, y_val, y_test

def main():
    config = Config()
    labels=config.LABELS_EMOTION
    labels_text=config.LABELS_EMOTION_TEXT
    labels_text = convert_to_int_keys(labels_text)
    
    print(labels, labels_text)
    
    
if __name__ == "__main__":
    
    main()

    # str_dict = {'1': 'one', '2': 'two', '3': 'three'}
    # print("Original string key dictionary:", str_dict)
    # swapped_str_dict = swap_keys_values(str_dict, 'str')
    # print("Swapped string key dictionary:", swapped_str_dict)

    # print()

    # int_dict = {1: 'one', 2: 'two', 3: 'three'}
    # print("Original integer key dictionary:", int_dict)
    # swapped_int_dict = swap_keys_values(int_dict, 'int')
    # print("Swapped integer key dictionary:", swapped_int_dict)
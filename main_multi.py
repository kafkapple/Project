

# # common
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader, random_split
# import moviepy.editor as mp

# # text
# from sentence_transformers import SentenceTransformer

# # speech
# import torchaudio
# from transformers import Wav2Vec2Processor, Wav2Vec2Model

from config import Config
from data_utils import load_data
import pandas as pd
import os
import numpy as np
import moviepy.editor as mp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# speech
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model

RANDOM_STATE = 2024
# fetch file paths and labels for speech data
def preprocess_data_meld(data_dir, text_train_df):
    data = []
    labels = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                data.append(file_path)

                dialogue_id = int(file.split("_")[0][3:])
                utterance_id = int(file.split("_")[1].split(".")[0][3:])
                label = text_train_df.loc[(text_train_df["Dialogue_ID"]==dialogue_id) & (text_train_df["Utterance_ID"]==utterance_id), "Emotion"].item()
                labels.append(label)

    if len(data) == 0:
        raise ValueError("No valid .wav files found in the dataset.")
    return np.array(data), np.array(labels)
# wav2vec

# Import the Wav2Vec2Processor class from the transformers library using a pre-trained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# Import the Wav2Vec2Model class from the transformers library using a pre-trained model
wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Define a function to extract features from the waveform with a given sample rate
def extract_speech_features(waveform, sample_rate):

    # Initialize an empty list to store the features
    features = []

    # Check if the waveform has more than one channel (stereo), if so, convert it to mono by averaging the channels
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)

    # If the sample rate of the waveform is not 16000 Hz, resample it to 16000 Hz
    if sample_rate != 16000:

        # Create a resampler object to convert the waveform's sample rate to 16000 Hz
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)

        # Apply the resampler to the waveform
        waveform = resampler(waveform)

    # Process the waveform to prepare it for the Wav2Vec2 model, converting it to tensors and padding if necessary
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    # Disable gradient calculation to save memory and computation
    with torch.no_grad():
        # Pass the processed inputs through the Wav2Vec2 model to get the output features
        outputs = wav2vec2_model(**inputs)

    # Extract the last hidden state from the model outputs, squeeze out unnecessary dimensions, and average the features
    wav2vec2_features = outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()

    # Append the extracted features to the features list
    features.append(wav2vec2_features)

    # Return the list of features
    return features

class SpeechDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = self.data[idx]
        label = self.labels[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        features = extract_speech_features(waveform, sample_rate)
        return features, label

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
        source_path = os.path.join(config.DATA_DIR, 'MELD', TARGET, TARGET_SPLIT, f'dia{dialogue_id}_utt{utterance_id}.mp4')
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

def data_prep(df, target, n_samples=0):
    classes = df[target].unique()
    n_class=len(classes)
    df_c= df.copy()
    print(f'Total number of class: {n_class} - {classes}')
    if n_samples !=0:
        n_samples_per_class = int(n_samples/n_class)
        print(f'{n_samples} per class will be sampled')
        df_c = df_c.groupby(target, group_keys=False).apply(lambda x: x.sample(n_samples_per_class))
    print(f"\nCounts for {target} each classes:")
    print(df_c[target].value_counts())
    return df_c


def main():
    config = Config()
    config.select_dataset = 'MELD'
    # ### Dataset
    # data_dir = load_data(config)#, config.dataset['MELD'])
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_samples=200
        # text data
    df = pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv')
    
    df_sampled = data_prep(df, 'Emotion', n_samples=n_samples)
    df.to_csv(os.path.join(config.DATA_DIR, 'MELD_train.csv'))
    df_sampled.to_csv(os.path.join(config.DATA_DIR, f'MELD_train_sampled_toy.csv'))
    
    
    
# save audio files corresponding to Dialogue_ID and Utterance_ID in text_train_df_toy
    data_dir = os.path.join(config.DATA_DIR, 'MELD', f'train_audio_toy')
    prep_audio(config, df_sampled, data_dir, 'train')
    #data, labels = preprocess_data_meld(data_dir, df_sampled)
    
    
  # 2
    #speech_data, speech_labels = preprocess_data(data_dir, text_train_df_toy)
    #speech_dataset = SpeechDataset(speech_data, speech_labels)
# # text data
# text_train_df = pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv')
# text_train_df_toy = text_train_df.sample(100, random_state=RANDOM_STATE, ignore_index=True)

# # speech data -- run only once
# !wget https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
# !tar -xvzf MELD.Raw.tar.gz
# !tar -xvzf MELD.Raw/train.tar.gz
# import os

if __name__ == "__main__":
    main()

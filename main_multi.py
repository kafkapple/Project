

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

RANDOM_STATE = 2024
# fetch file paths and labels for speech data
def preprocess_data(data_dir, text_train_df):
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
from tqdm import tqdm
def main():
    config = Config()
    # config.select_dataset = 'MELD'
    # ### Dataset
    # data, labels = load_data(config)#, config.dataset['MELD'])
    # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        # text data
    text_train_df = pd.read_csv('https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv')
    text_train_df_toy = text_train_df.sample(100, random_state=RANDOM_STATE, ignore_index=True)
    
    
# save audio files corresponding to Dialogue_ID and Utterance_ID in text_train_df_toy
    dest_base = os.path.join(config.DATA_DIR, 'MELD', 'train_audio')
    os.makedirs(dest_base, exist_ok=True)
    for dialogue_id, utterance_id in tqdm(zip(text_train_df_toy["Dialogue_ID"], text_train_df_toy["Utterance_ID"])):
      # Source and destination paths
      source_path =      os.path.join(config.DATA_DIR, 'MELD',  'train','train_splits', f'dia{dialogue_id}_utt{utterance_id}.mp4')
      destination_path = os.path.join(dest_base, f'dia{dialogue_id}_utt{utterance_id}.wav') #f'/content/drive/MyDrive/NMA/Upbeat-Tuberose-Poplar/data/raw/MELD_toy/    

      #print(f'Source to Dest: {source_path} -> {destination_path}')
      # Load the video file
      video = mp.VideoFileClip(source_path)

      # Extract the audio
      audio = video.audio

      # Save the audio file
      audio.write_audiofile(destination_path)
  
  # 2
    # data_dir = '/content/drive/MyDrive/NMA/Upbeat-Tuberose-Poplar/data/raw/MELD_toy/'
    # speech_data, speech_labels = preprocess_data(data_dir, text_train_df_toy)

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

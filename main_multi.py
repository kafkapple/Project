

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

def main():
    config = Config()
    config.select_dataset = 'MELD'
    ### Dataset
    data, labels = load_data(config)#, config.dataset['MELD'])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  

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

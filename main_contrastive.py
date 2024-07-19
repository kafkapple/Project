import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, Wav2Vec2Model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 임베딩 모델 정의
class TextEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.fc = nn.Linear(768, 128)  # 임베딩 차원을 128로 줄임

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])

class AudioEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.fc = nn.Linear(768, 128)  # 임베딩 차원을 128로 줄임

    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        return self.fc(outputs.last_hidden_state.mean(dim=1))

# Triplet Loss 정의
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

# 데이터셋 및 데이터 로더 (예시)
class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]

        # 같은 클래스에서 positive 샘플 선택
        positive_idx = np.random.choice(np.where(self.labels == anchor_label)[0])
        positive = self.data[positive_idx]

        # 다른 클래스에서 negative 샘플 선택
        negative_label = np.random.choice(list(set(self.labels) - {anchor_label}))
        negative_idx = np.random.choice(np.where(self.labels == negative_label)[0])
        negative = self.data[negative_idx]

        return anchor, positive, negative, anchor_label

# 학습 함수
def train_embedding(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for anchor, positive, negative, _ in dataloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            loss = criterion(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

# 임베딩 시각화 함수
def visualize_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for data, _, _, label in dataloader:
            data = data.to(device)
            embed = model(data).cpu().numpy()
            embeddings.append(embed)
            labels.append(label.numpy())
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title("Embeddings Visualization (t-SNE)")
    plt.show()

# 메인 실행 코드
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 텍스트 모델
    text_model = TextEmbeddingModel().to(device)
    text_criterion = TripletLoss()
    text_optimizer = optim.Adam(text_model.parameters(), lr=1e-5)

    # 오디오 모델
    audio_model = AudioEmbeddingModel().to(device)
    audio_criterion = TripletLoss()
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=1e-5)

    # 데이터 로더 (예시, 실제 데이터에 맞게 수정 필요)
    text_data = torch.randn(1000, 512)  # 임의의 텍스트 데이터
    audio_data = torch.randn(1000, 16000)  # 임의의 오디오 데이터
    labels = torch.randint(0, 6, (1000,))  # 6개 클래스

    text_dataset = TripletDataset(text_data, labels)
    audio_dataset = TripletDataset(audio_data, labels)

    text_loader = DataLoader(text_dataset, batch_size=32, shuffle=True)
    audio_loader = DataLoader(audio_dataset, batch_size=32, shuffle=True)

    # 학습
    print("Training Text Embedding Model")
    train_embedding(text_model, text_loader, text_criterion, text_optimizer, device, num_epochs=10)

    print("Training Audio Embedding Model")
    train_embedding(audio_model, audio_loader, audio_criterion, audio_optimizer, device, num_epochs=10)

    # 시각화
    print("Visualizing Text Embeddings")
    visualize_embeddings(text_model, text_loader, device)

    print("Visualizing Audio Embeddings")
    visualize_embeddings(audio_model, audio_loader, device)

if __name__ == "__main__":
    main()
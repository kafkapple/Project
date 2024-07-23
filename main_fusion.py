import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, RobertaModel

class MultiModalFusion(nn.Module):
    def __init__(self, wav2vec_path, roberta_path, num_classes):
        super().__init__()
        # 파인튜닝된 wav2vec2 모델 로드
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path)
        # 파인튜닝된 RoBERTa 모델 로드
        self.roberta_model = RobertaModel.from_pretrained(roberta_path)
        
        wav2vec_dim = self.wav2vec_model.config.hidden_size
        roberta_dim = self.roberta_model.config.hidden_size
        
        self.attention = nn.MultiheadAttention(embed_dim=wav2vec_dim + roberta_dim, num_heads=8)
        self.fc = nn.Linear(wav2vec_dim + roberta_dim, num_classes)
    
    def forward(self, audio_input, text_input):
        # wav2vec2 모델로 오디오 특징 추출
        audio_features = self.wav2vec_model(audio_input).last_hidden_state
        # RoBERTa 모델로 텍스트 특징 추출
        text_features = self.roberta_model(**text_input).last_hidden_state
        
        # 특징 결합
        combined_features = torch.cat((audio_features, text_features), dim=-1)
        
        # 어텐션 적용
        attn_output, _ = self.attention(combined_features, combined_features, combined_features)
        
        # 전역 평균 풀링
        pooled_features = attn_output.mean(dim=1)
        
        # 최종 분류
        output = self.fc(pooled_features)
        
        return output

# 모델 사용 예시
wav2vec_path = "./wav2vec2_finetuned"  # 파인튜닝된 wav2vec2 모델 경로
roberta_path = "./roberta_finetuned"   # 파인튜닝된 RoBERTa 모델 경로
model = MultiModalFusion(wav2vec_path, roberta_path, num_classes=8)
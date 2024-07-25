

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Model, Wav2Vec2ForSequenceClassification
import os
# from config import Config # custom modeul

def print_model_info(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Structure:")
    print(model)
    print("\nLayer-wise details:")
    
    for name, module in model.named_modules():
        if not list(module.children()):  
            print(f"\nLayer: {name}")
            print(f"Type: {type(module).__name__}")
            params = sum(p.numel() for p in module.parameters())
            print(f"Parameters: {params:,}")

    print(f"\nTotal layers: {len(list(model.modules()))}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
def get_embeddings(model, data_loader):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, batch_labels = batch
            _ = model(inputs)  # forward pass
            batch_embeddings = model.get_penultimate_features()
            
            embeddings.append(batch_embeddings)
            labels.extend(batch_labels)
    
    return torch.cat(embeddings), labels

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

def freeze_wav2vec(model):
    for param in model.wav2vec.parameters():
        param.requires_grad = False

def unfreeze_layers(model, num_layers):
    for param in model.parameters():
        param.requires_grad = False
    
    all_modules = list(model.modules())
    
    for module in all_modules[-num_layers:]:
        for param in module.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    
### model class declaration
class EmotionRecognitionBase(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = self._get_activation(activation)
        self.input_size = input_size
        self.num_classes = num_classes
    
    def _get_activation(self, activation):
        if activation == 'gelu':
            return nn.GELU()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

class EmotionRecognitionWithWav2Vec(nn.Module):
    def __init__(self, num_classes, config, dropout_rate=0.5, activation='relu', use_wav2vec=True, input_size=None):
        super().__init__()
        
        self.use_wav2vec = use_wav2vec
        self.config = config
        self.penultimate_features = None
        
        if use_wav2vec:
            self.wav2vec = Wav2Vec2Model.from_pretrained(self.config.path_pretrained)
            self.wav2vec.config.mask_time_length = config.mask_time_length
            wav2vec_output_size = self.wav2vec.config.hidden_size
        else:
            wav2vec_output_size = input_size
        
        self.emotion_classifier = EmotionRecognitionModel_v2(
            input_size=wav2vec_output_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        # Register hook to capture penultimate layer output
        self.emotion_classifier.fc3.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.penultimate_features = output

    def forward(self, input_values):
        if self.use_wav2vec:
            if input_values.dim() == 4:
                input_values = input_values.squeeze(2)
            if input_values.dim() == 3:
                input_values = input_values.squeeze(1)
            wav2vec_outputs = self.wav2vec(input_values, output_hidden_states=True).last_hidden_state
            features = torch.mean(wav2vec_outputs, dim=1)
        else:
            features = input_values.view(input_values.size(0), -1)
        
        emotion_logits = self.emotion_classifier(features)
        return emotion_logits

    def get_penultimate_features(self):
        return self.penultimate_features

class EmotionRecognitionModel_v2(EmotionRecognitionBase):
    def __init__(self, input_size, num_classes, dropout_rate, activation):
        super().__init__(input_size, num_classes, dropout_rate, activation)
        momentum = 0.01
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=momentum)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, momentum=momentum)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64, momentum=momentum)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.activation(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)
    
def main():
    #config = Config()
    DATA_NAME='MELD'
    if DATA_NAME=='MELD':
        # config.LABELS_EMOTION =config.LABELS_EMO_MELD # emotion class
        LABELS_EMOTION = {
        0: 'anger', 1: 'disgust', 2: 'fear', 3: 'joy',
        4: 'neutral', 5: 'sadness', 6: 'surprise'
    }
    else:
        LABELS_EMOTION = {}
    # config.path_pretrained = os.path.join(MODEL_BASE_DIR, 'finetuned', 'wav2vec_finetuned_best') 
    
    path = os.path.join(os.getcwd(), 'Project_NMA', 'models', 'finetuned', 'wav2vec_finetuned_best')#os.path.join('')# model path
    
    # Model load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. wav2vec only
    model = Wav2Vec2ForSequenceClassification.from_pretrained(path, num_labels=len(LABELS_EMOTION), output_hidden_states=True)
    model.to(device)
    model = load_model(model, path)
    print_model_info(model)
    print('Model is loaded.')
    
    # 2. wav2vec + classifier
    # model= EmotionRecognitionWithWav2Vec(num_classes=len(config.LABELS_EMOTION), config=config,  input_size=train_loader.dataset[0][0].shape[1], dropout_rate=config.DROPOUT_RATE,
    #     activation=config.ACTIVATION, use_wav2vec=True) 
    # model.to(device)
    
    # freeze_wav2vec(model)
    # n_unfreeze=10
    # unfreeze_layers(model, n_unfreeze)
    
    
    
if __name__ == '__main__':
    main()
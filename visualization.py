import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import wandb
import torch
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
import os
from scipy.spatial.distance import cosine

import torch.nn.functional as F

from data_utils import get_logits_from_output

def compute_layer_similarity(activations, device):
    n_layers = len(activations)
    similarity_matrix = torch.zeros((n_layers, n_layers), device=device)
    
    for i in range(n_layers):
        for j in range(n_layers):
            flattened_i = activations[i].view(activations[i].size(0), -1)
            flattened_j = activations[j].view(activations[j].size(0), -1)
            
            # 두 텐서의 차원을 맞추기 위해 더 작은 차원을 가진 텐서에 맞춰 조정
            min_dim = min(flattened_i.size(1), flattened_j.size(1))
            flattened_i = flattened_i[:, :min_dim]
            flattened_j = flattened_j[:, :min_dim]
            
            # 각 샘플 쌍에 대해 cosine similarity 계산
            cos_sim = F.cosine_similarity(flattened_i.unsqueeze(1), flattened_j.unsqueeze(0), dim=2)
            # 모든 샘플 쌍의 평균 similarity
            similarity_matrix[i, j] = cos_sim.mean()
    
    return similarity_matrix.cpu().numpy()
      # flattened_i.unsqueeze(1)의 결과 shape: (batch_size, 1, flattened_features)
            #flattened_j.unsqueeze(0)의 결과 shape: (1, batch_size, flattened_features)
            #브로드캐스팅PyTorch는 이 두 텐서를 자동으로 브로드캐스팅하여 다음과 같은 형태로 확장합니다: 두 텐서 모두 (batch_size, batch_size, flattened_features) 형태로 확장
            #입력2: (batch_size, batch_size, flattened_features)
            #dim=2: 마지막 차원(특성 차원)을 따라 코사인 유사도를 계산합니다.


import torch
from collections import OrderedDict

def get_layer_activations(model, inputs, num_layers_from_end=None):
    activations = OrderedDict()
    handles = []

    def hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                activations[name] = output[0].detach()
            elif hasattr(output, 'last_hidden_state'):
                activations[name] = output.last_hidden_state.detach()
            elif isinstance(output, SequenceClassifierOutput):
                activations[name] = output.logits.detach()
            else:
                print(f"Warning: Unexpected output type for layer {name}: {type(output)}")
        return hook_fn

    try:
        # 모든 모듈에 대해 훅 등록
        for name, module in model.named_modules():
            handles.append(module.register_forward_hook(hook(name)))

        # 모델 실행
        with torch.no_grad():
            outputs = model(inputs)

        # 모델의 최종 출력 처리
        if isinstance(outputs, SequenceClassifierOutput):
            activations['model_output'] = outputs.logits.detach()
        elif isinstance(outputs, torch.Tensor):
            activations['model_output'] = outputs.detach()
        else:
            print(f"Warning: Unexpected model output type: {type(outputs)}")

    finally:
        # 모든 훅 제거
        for handle in handles:
            handle.remove()

    # 끝에서부터 원하는 개수의 레이어만 선택
    if num_layers_from_end is not None:
        total_layers = len(activations)
        start_index = max(0, total_layers - num_layers_from_end)
        selected_activations = OrderedDict(list(activations.items())[start_index:])
    else:
        selected_activations = activations

    return selected_activations

# def get_all_layer_activations(model, inputs):
#     activations = OrderedDict()
#     handles = []

#     def hook(name):
#         def hook_fn(module, input, output):
#             if isinstance(output, torch.Tensor):
#                 activations[name] = output.detach()
#             elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
#                 activations[name] = output[0].detach()
#             elif hasattr(output, 'last_hidden_state'):
#                 activations[name] = output.last_hidden_state.detach()
#             elif isinstance(output, SequenceClassifierOutput):
#                 activations[name] = output.logits.detach()
#             else:
#                 print(f"Warning: Unexpected output type for layer {name}: {type(output)}")
#         return hook_fn

#     try:
#         # 모든 모듈에 대해 훅 등록
#         for name, module in model.named_modules():
#             handles.append(module.register_forward_hook(hook(name)))

#         # 모델 실행
#         with torch.no_grad():
#             outputs = model(inputs)

#         # 모델의 최종 출력 처리
#         if isinstance(outputs, SequenceClassifierOutput):
#             activations['model_output'] = outputs.logits.detach()
#         elif isinstance(outputs, torch.Tensor):
#             activations['model_output'] = outputs.detach()
#         else:
#             print(f"Warning: Unexpected model output type: {type(outputs)}")

#     finally:
#         # 모든 훅 제거
#         for handle in handles:
#             handle.remove()

#     return activations

def perform_rsa(model, data_loader, device):
    model.eval()
    all_activations = {}
    num_layers_from_end = 5

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Collecting activations"):
            inputs = batch['audio'].to(device)
            batch_activations = get_layer_activations(model, inputs, num_layers_from_end)
            all_activations.append(batch_activations)
      
    # 모든 배치의 활성화를 결합
    combined_activations = {name: torch.cat(acts, dim=0) for name, acts in all_activations.items() if acts}
    
    # combined_activations의 형태 확인
    for name, act in combined_activations.items():
        print(f"Layer {name} activation shape: {act.shape}")

    layer_names = list(combined_activations.keys())
    activations_list = [combined_activations[name] for name in layer_names]

    # compute_layer_similarity 함수 사용
    similarity_matrix = compute_layer_similarity(activations_list, device)

    # 시각화
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, xticklabels=layer_names, yticklabels=layer_names, cmap='coolwarm')
    plt.title("Layer-wise Representation Similarity Analysis")
    plt.xlabel("Layers")
    plt.ylabel("Layers")
    plt.tight_layout()
    
    return plt.gcf()

def compute_similarity(act1, act2):
    # 활성화를 2D로 펼치기
    flat1 = act1.view(act1.size(0), -1)
    flat2 = act2.view(act2.size(0), -1)
    
    # 코사인 유사도 계산
    similarity = F.cosine_similarity(flat1.unsqueeze(1), flat2.unsqueeze(0), dim=2)
    
    # 평균 유사도 반환
    return similarity.mean().item()
    
def save_and_log_figure(stage, fig, config, name, title):
    """Save figure to file and log to wandb"""
    fig_path = os.path.join(config.MODEL_RESULTS, f"{name}_{config.global_epoch}.png")
    fig.savefig(fig_path)
    wandb.log({stage:{f"{name}": wandb.Image(fig_path, caption=title)}}, step=config.global_epoch)
    plt.close(fig)
    
def visualize_results(config, model, data_loader, device, log_data, stage):
    print('\nVisualization of results starts.\n')

    # Confusion Matrix and Embeddings visualization for all stages
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Preparing data for Visualizing..."):
            if isinstance(batch, dict):
                inputs = batch['audio'].to(device)
                labels = batch['label'].to(device)
            else:  # batch가 튜플인 경우
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)  # labels도 device로 이동
            outputs = model(inputs)
            
            #activations = get_all_layer_activations(model, inputs)
            try:
                logits = get_logits_from_output(outputs)
            except Exception as e:
                print(f'Error in get_logits_from_output during evaluation: {e}')
                logits = outputs  #
            
            if logits is None:
                raise ValueError("Unable to extract logits from model output")
            
            # Penultimate features 추출 (가능한 경우)
                       
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                penultimate_features = outputs.hidden_states[-2]
            elif hasattr(model, 'get_penultimate_features'):
                penultimate_features = model.get_penultimate_features()
            else:
                print("Warning: Hidden states not available. Using logits as embeddings.")
                penultimate_features = logits
            

            _, preds = torch.max(logits, 1) 
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if len(penultimate_features.shape) > 2:
                penultimate_features = penultimate_features.mean(dim=1)  # 시퀀스 차원에 대해 평균 계산
            all_embeddings.extend(penultimate_features.cpu().numpy())
            
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)

    # Confusion Matrix
    fig_cm = plot_confusion_matrix(all_labels, all_preds, config.LABELS_EMOTION)
    save_and_log_figure(stage, fig_cm, config, "confusion_matrix", f"{stage} Confusion Matrix")

    # Embeddings visualization
    max_samples = config.N_EMBEDDINGS # to show
    
    if len(all_embeddings) > max_samples:
        indices = np.random.choice(len(all_embeddings), max_samples, replace=False)
        all_embeddings = all_embeddings[indices]
        all_labels = all_labels[indices]
    
    try:
        fig_embd = visualize_embeddings(config, all_embeddings, all_labels)
    
        save_and_log_figure(stage, fig_embd, config, "embeddings", f"{stage.capitalize()} Embeddings (t-SNE)")
    except:
        print('No embedding.')    
    try:
        fig_rsa = perform_rsa(model, data_loader, config.device)
        save_and_log_figure(stage, fig_rsa, config, "Representation_similarity", f"{stage.capitalize()}")
        plt.close(fig_rsa)
    except:
        print('RSA error')
    if stage in ['train', 'val'] and log_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        try:
            epochs = [entry['epoch'] for entry in log_data]
            losses = [entry['loss'] for entry in log_data[stage]]
            accuracies = [entry['accuracy'] for entry in log_data[stage]]
            
            ax1.plot(epochs, losses, 'bo-')
            ax1.set_title(f'{stage.capitalize()} Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            
            ax2.plot(epochs, accuracies, 'ro-')
            ax2.set_title(f'{stage.capitalize()} Accuracy')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            
            save_and_log_figure(stage, fig, config, "learning_curves", f"{stage.capitalize()} Learning Curves")
        except:
            print('Err. no learning curve.')
    
def visualize_embeddings(config, embeddings, labels, method='tsne'):
    print('\nVisualization of embedding starts...\n')
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Invalid method. Use 'pca' or 'tsne'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    string_labels = [config.LABELS_EMOTION.get(int(label), f"Unknown_{label}") for label in labels]

    #string_labels = [config.LABELS_EMOTION.get(str(int(label)), f"Unknown_{label}") for label in labels]
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': string_labels
    })

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette="deep", legend="full", ax=ax)
    ax.set_title(f"{method.upper()} of Emotion Recognition Embeddings")
    
    return fig


def plot_learning_curves(config):
    # Assuming we've saved loss and accuracy values during training
    train_losses = np.load(f"{config.MODEL_DIR}/train_losses.npy")
    val_losses = np.load(f"{config.MODEL_DIR}/val_losses.npy")
    train_accuracies = np.load(f"{config.MODEL_DIR}/train_accuracies.npy")
    val_accuracies = np.load(f"{config.MODEL_DIR}/val_accuracies.npy")
    
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    ax2.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    #plt.tight_layout()
    img_path=f"{config.MODEL_RESULTS}/learning_curves_{config.global_epoch}.png"
    plt.savefig(img_path)
    wandb.log({"learning_curves": wandb.Image(img_path)})
    
def plot_confusion_matrix(labels, preds, labels_emotion, normalize=True):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels_emotion.values(), yticklabels=labels_emotion.values(), ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    return fig

def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    epochs = range(1, len(history['train']['loss']) + 1)
    
    ax1.plot(epochs, history['train']['loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val']['loss'], 'ro-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(epochs, history['train']['accuracy'], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, history['val']['accuracy'], 'ro-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    #plt.tight_layout()
    return fig


    
# def extract_embeddings_and_predictions(model, data_loader, device):
#     model.eval()
#     all_embeddings = []
#     all_labels = []
#     all_predictions = []
    
#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs = inputs.to(device)
#             outputs = model(inputs)
            
#             hidden_states = outputs.last_hidden_state
#             pooled_output = torch.mean(hidden_states, dim=1)
            
#             logits = model.classifier(pooled_output)
            
#             # predictions = outputs.argmax(dim=1).cpu().numpy()
            
#             all_embeddings.extend(pooled_output.cpu().numpy())
#             all_labels.extend(labels.numpy())
#             all_predictions.extend(labels.cpu().numpy())
    
#     return np.array(all_embeddings), np.array(all_labels), np.array(all_predictions)

# def perform_rsa(model, data_loader, device):
#     model.eval()
#     all_activations = []
#     labels = []
#     print('rsa1')
#     with torch.no_grad():
#         for batch in data_loader:
#             inputs = batch['audio'].to(device)
#             batch_labels = batch['label']
            
#             activations = get_all_layer_activations(model, inputs)
#             all_activations.append([act.cpu().numpy() for act in activations])
#             labels.extend(batch_labels.numpy())
#     print('rsa2')
#     # Combine activations from all batches
#     # 모든 배치의 활성화를 결합
#     combined_activations = [torch.cat([batch[i] for batch in all_activations]) for i in range(len(all_activations[0]))]
    
#     # combined_activations의 형태 확인
#     for i, act in enumerate(combined_activations):
#         print(f"Layer {i} activation shape: {act.shape}")

    
#     layer_similarity_matrix = compute_layer_similarity(combined_activations, device)
    
#     labels = np.array(labels)
#     label_matrix = np.equal.outer(labels, labels).astype(int)

#     # Plot the layer similarity matrix and the label correlation matrix
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

#     sns.heatmap(layer_similarity_matrix, cmap='coolwarm', ax=ax1)
#     ax1.set_title("Layer Similarity Matrix")
#     ax1.set_xlabel("Layers")
#     ax1.set_ylabel("Layers")

#     sns.heatmap(label_matrix, cmap='coolwarm', ax=ax2)
#     ax2.set_title("Label Correlation Matrix")

#     plt.suptitle("Layer-wise Representation Similarity Analysis", fontsize=16)
#     return fig




# def visualize_metric(model, data_loader):
#     fig = plot_confusion_matrix(all_labels, all_preds, config.LABELS_EMOTION)
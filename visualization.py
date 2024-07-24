import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import pandas as pd
from tqdm import tqdm
import os

import wandb

from collections import OrderedDict

from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
import torch

from data_utils import get_logits_from_output
import torch
import torch.nn.functional as F

def get_matching_layers(model, inputs, num_layers=5):
    activations = OrderedDict()
    
    def hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                activations[name] = output[0].detach()
        return hook_fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook(name)))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    # 페널티메이트 레이어 찾기
    penultimate_layer = list(activations.items())[-2]
    penultimate_shape = penultimate_layer[1].shape[1:]  # 배치 차원 제외

    matching_layers = OrderedDict()
    for name, act in reversed(list(activations.items())):
        if act.shape[1:] == penultimate_shape:
            matching_layers[name] = act
            if len(matching_layers) == num_layers:
                break

    print(f"Matching layers found: {list(matching_layers.keys())}")
    return matching_layers

def compute_layer_similarity(activations):
    layer_names = list(activations.keys())
    n_layers = len(layer_names)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    for i in range(n_layers):
        for j in range(n_layers):
            act_i = activations[layer_names[i]]
            act_j = activations[layer_names[j]]
            
            # 각 활성화를 2D로 평탄화
            flat_i = act_i.view(act_i.size(0), -1)
            flat_j = act_j.view(act_j.size(0), -1)
            
            
            print(f"Computing similarity between {layer_names[i]} and {layer_names[j]}")
            print(f"Shape of act_i: {act_i.shape}, Shape of act_j: {act_j.shape}")
            
            print(f"Shape after flattening - flat_i: {flat_i.shape}, flat_j: {flat_j.shape}")
            # 정규화
            norm_i = torch.norm(flat_i, p=2, dim=1, keepdim=True)
            norm_j = torch.norm(flat_j, p=2, dim=1, keepdim=True)
            flat_i_normalized = flat_i / (norm_i + 1e-8)
            flat_j_normalized = flat_j / (norm_j + 1e-8)
            # 코사인 유사도 계산
            similarity = torch.mm(flat_i_normalized, flat_j_normalized.t())
            
            # 평균 유사도 계산
            similarity_matrix[i, j] = similarity.mean().item()
    
    return similarity_matrix, layer_names

def perform_rsa(model, data_loader, device, num_layers=5):
    model.eval()
    
    for batch in data_loader:
        inputs = batch['audio'].to(device)
        if inputs.dim() == 4:
            inputs = inputs.squeeze(2)
        if inputs.dim() == 3:
            inputs = inputs.squeeze(1)
        
        print("Input shape:", inputs.shape)
        
        matching_activations = get_matching_layers(model, inputs, num_layers)
        break  # 첫 번째 배치만 사용

    print("Matching activations shapes:", {k: v.shape for k, v in matching_activations.items()})

    similarity_matrix, layer_names = compute_layer_similarity(matching_activations)

    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', 
                xticklabels=layer_names, yticklabels=layer_names)
    plt.title("Layer-wise Representation Similarity Analysis")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()

    return plt.gcf()

def save_and_log_figure(stage, fig, config, name, title):
    """Save figure to file and log to wandb"""
    fig_path = os.path.join(config.MODEL_RESULTS, f"{name}_{config.global_epoch}.png")
    fig.savefig(fig_path)
    wandb.log({stage:{f"{name}": wandb.Image(fig_path, caption=title)}}, step=config.global_epoch)
    plt.close(fig)
    
def visualize_results(config, model, data_loader, device, log_data, stage):
    print('\n[Visualization]\n')

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

    # Confusion Matrix
    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_embeddings = np.array(all_embeddings)
    
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
    
        save_and_log_figure(stage, fig_embd, config, "Embeddings", f"{stage.capitalize()} Embeddings (t-SNE)")
    except:
        print('No embedding.')   
         
    try:
        fig_rsa = perform_rsa(model, data_loader, config.device)
        save_and_log_figure(stage, fig_rsa, config, "Representation similarity", f"{stage.capitalize()}")
        
    except:
        print('(Err)RSA\n')
    
    try:
        fig_metric = plot_learning_curves(log_data)
        save_and_log_figure(stage, fig_metric, config, "Learning curve", f"{stage.capitalize()}")
    except:
        print('(Err) learning curve\n')
    
def visualize_embeddings(config, embeddings, labels, method='tsne'):
    print('\n\nVisualization of embedding...\n')
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

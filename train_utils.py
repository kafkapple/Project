import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
from visualization import visualize_results#plot_confusion_matrix, 
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for features, batch_labels in progress_bar:
        features, batch_labels = features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    metric_average='weighted'
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=metric_average)
    recall = recall_score(all_labels, all_preds, average=metric_average)
    f1 = f1_score(all_labels, all_preds, average=metric_average)
    
    return epoch_loss, accuracy, precision, recall, f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, batch_labels in tqdm(dataloader, desc="Evaluating"):
            features, batch_labels = features.to(device), batch_labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return epoch_loss, accuracy, precision, recall, f1, all_labels, all_preds

def train_model(model, train_loader, val_loader, config, device, optimizer, criterion):
    best_val_accuracy = 0
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }

    for epoch in range(config.NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update history
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            history['train'][metric].append(train_metrics[i])
            history['val'][metric].append(val_metrics[i])
        
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")
        print(f"Train - Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}, F1: {train_metrics[4]:.4f}")
        print(f"Val - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}, F1: {val_metrics[4]:.4f}")
        
        # Log metrics
        log_metrics('train', train_metrics, epoch)
        log_metrics('val', val_metrics[:5], epoch)  # val_metrics might have 7 values, we only need first 5
        
        if val_metrics[1] > best_val_accuracy:  # val_metrics[1] is accuracy
            best_val_accuracy = val_metrics[1]
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Best model saved to {config.MODEL_SAVE_PATH}")
        
        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_accuracy': best_val_accuracy,
            'id_wandb': config.id_wandb
        }
        torch.save(ckpt, config.CKPT_SAVE_PATH)
        print(f"Checkpoint saved to {config.CKPT_SAVE_PATH} at epoch {epoch+1}\n{ckpt['id_wandb']}\n")
        

    return history, best_val_accuracy

def log_metrics(stage, stage_metrics, epoch):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    log_dict = {
        stage: {metric: value for metric, value in zip(metrics, stage_metrics)},
        'epoch': epoch
    }
    wandb.log(log_dict, step=epoch)
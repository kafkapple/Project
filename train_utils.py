import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb


def load_checkpoint(ckpt_path, model, optimizer, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  #

    best_val_accuracy = checkpoint['best_val_accuracy']
    id_wandb = checkpoint['id_wandb']
    return model, optimizer, start_epoch, best_val_accuracy, id_wandb

def train_model(model, train_loader, val_loader, config, device, optimizer, criterion):
    best_val_accuracy = 0
    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }

    start_epoch = config.initial_epoch  # config에서 initial_epoch를 가져옵니다

    for epoch in range(start_epoch, start_epoch + config.NUM_EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update history
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            history['train'][metric].append(train_metrics[i])
            history['val'][metric].append(val_metrics[i])
        
        print(f"Epoch [{epoch}/{start_epoch + config.NUM_EPOCHS - 1}]")
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
            'id_wandb': wandb.run.id
        }
        torch.save(ckpt, config.CKPT_SAVE_PATH)

        print(f"Checkpoint saved to {config.CKPT_SAVE_PATH} at epoch {epoch+1}\n{ckpt['id_wandb']}\n")
        

    return history, best_val_accuracy
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
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name}: grad_norm: {param.grad.norm().item()}")
    
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

def log_metrics(stage, stage_metrics, epoch):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    log_dict = {
        stage: {metric: value for metric, value in zip(metrics, stage_metrics)},
        'epoch': epoch
    }
    wandb.log(log_dict, step=epoch)
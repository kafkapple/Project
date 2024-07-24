import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import os
from collections import namedtuple
from visualization import visualize_results


def evaluate_baseline(model, X_test, y_test, config):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=config.METRIC_AVG)
    recall = recall_score(y_test, y_pred, average=config.METRIC_AVG)
    f1 = f1_score(y_test, y_pred, average=config.METRIC_AVG)
    log_loss = (y_test, y_pred)
    return log_loss, accuracy, precision, recall, f1, y_pred


def load_checkpoint(config, model, optimizer, device):
    ckpt_path=config.CKPT_SAVE_PATH
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loading Optimizer info. ')
        if config.SCHEDULER: # chk
            print('Scheduler on.')
            ### Scheduler    
            T_max = config.NUM_EPOCHS  *2
            eta_min = config.eta_min
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
            
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print('Loading Scheduler info.')
            
        
        global_epoch=checkpoint['global_epoch']

        print(f'Previously, total number of epoch: {global_epoch} was trained.')
              #\nStart from epoch: {start_epoch}\n')

        best_val_loss = checkpoint['best_val_loss']
        try:
            config.id_wandb = checkpoint['id_wandb']
        except:
            print('no wand id.')   
        try:
            config.sweep_id = checkpoint['sweep_id']
        except:
            print('There is a Checkpoint, but No sweep data is found.')
            
        return model, optimizer, global_epoch, best_val_loss, config.id_wandb
    else:
        print("No checkpoint found.")
        return model, optimizer, 0, 0, wandb.util.generate_id()
    

def train_model(model, train_loader, val_loader, config, device, optimizer, criterion):
    best_val_loss = 1000
    best_val_acc=0.0
    early_stop_counter = 0

    history = {
        'train': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    }

    global_epoch = config.global_epoch#config.initial_epoch
    start_epoch = global_epoch
    end_epoch=config.global_epoch + config.NUM_EPOCHS
    
    if config.SCHEDULER:
        T_max = config.NUM_EPOCHS *2
        eta_min = config.eta_min 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    progress_bar = tqdm(range(start_epoch+1, end_epoch+1), desc="[ Total Epoch Progress ]")

    for epoch in progress_bar:
        global_epoch+=1
        config.global_epoch=global_epoch
        #print(f'global epoch updated: {global_epoch}')
        train_metrics = train_epoch(config, model, train_loader, criterion, optimizer, device) #train
        val_metrics = evaluate_model(config, model, val_loader, criterion, device) #val
        # Update history
        for i, metric in enumerate(['loss', 'accuracy', 'precision', 'recall', 'f1']):
            history['train'][metric].append(train_metrics[i])
            history['val'][metric].append(val_metrics[i])
        
        print(f"Train - Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}, F1: {train_metrics[4]:.4f}")
        print(f"Val - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}, F1: {val_metrics[4]:.4f}")
        
        
        ######
        #Log metrics chk
        log_metrics('train', train_metrics, global_epoch)
        log_metrics('val', val_metrics[:5], global_epoch)  # val_metrics might have 7 values, we only need first 5
        
        if global_epoch % config.N_STEP_FIG ==0: # visualization for val data 
            visualize_results(config, model, val_loader, device, history, 'val')
            
        if config.SCHEDULER:
            scheduler.step()#val_metrics[0]) #[0] val loss
        
        print(f'Val loss/Best val loss:{val_metrics[0]:.4f}/{best_val_loss:.4f}')
        if val_metrics[0] < best_val_loss: #accuracy:  # val_metrics[1] is accuracy
            best_val_loss = val_metrics[0]
            # torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            # print(f"Best model saved to {config.MODEL_SAVE_PATH}")
            early_stop_counter = 0 # reset
        else:
            early_stop_counter+=1
        print(f'Val acc/Best val acc:{val_metrics[1]:.4f}/{best_val_acc:.4f}')
        if val_metrics[1] > best_val_acc:
            best_val_acc = val_metrics[1]
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New Best Model with higher accuracy found.\nBest model saved to {config.MODEL_SAVE_PATH}")
            
        # Save checkpoint
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'id_wandb': wandb.run.id,
            'global_epoch': global_epoch
        }
        if config.IS_SWEEP:
            print(f'Sweep is finished. ID is saved: {config.sweep_id}')
            ckpt['sweep_id']=config.sweep_id
        if config.SCHEDULER: #chk
            ckpt['scheduler_state_dict']= scheduler.state_dict()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")
            wandb.log({"learning_rate": current_lr}, step=global_epoch)
            
        torch.save(ckpt, config.CKPT_SAVE_PATH)

        print(f"Checkpoint saved to {config.CKPT_SAVE_PATH} at global epoch: {global_epoch}\n{ckpt['id_wandb']}\n")
            
        
        if early_stop_counter >= config.early_stop_epoch:
            print("Early Stopping!")
            break
    
    return history, best_val_loss, best_val_acc
def train_epoch(config, model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        features = batch['audio']
        batch_labels = batch['label']
        if config.VISUALIZE:
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {batch_labels.shape}")
            
        features, batch_labels = features.to(device), batch_labels.to(device)
        
        if config.VISUALIZE:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.BatchNorm1d):
                    print(f"{name} - mean: {module.running_mean.mean().item():.4f}, var: {module.running_var.mean().item():.4f}")
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        
        if config.GRADIENT_CLIP:
            if config.VISUALIZE:
                print('\nGradient Clipping')
            #gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    metric_average=config.METRIC_AVG
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=metric_average)
    recall = recall_score(all_labels, all_preds, average=metric_average)
    f1 = f1_score(all_labels, all_preds, average=metric_average)
    
    ### for debugging - gradient chk
    if config.VISUALIZE:
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad_norm: {param.grad.norm().item()}")
    
    return epoch_loss, accuracy, precision, recall, f1

def evaluate_model(config, model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    EvaluationResult = config.EvaluationResult
    
    with torch.no_grad():
        for batch in dataloader:
            features = batch['audio']
            batch_labels = batch['label']
            
            if config.VISUALIZE:
                print(f"Features shape: {features.shape}")
                print(f"Labels shape: {batch_labels.shape}")

        # for features, batch_labels in tqdm(dataloader, desc="Evaluating"):
            features, batch_labels = features.to(device), batch_labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=config.METRIC_AVG, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=config.METRIC_AVG)
    f1 = f1_score(all_labels, all_preds, average=config.METRIC_AVG)
    
    return EvaluationResult(epoch_loss, accuracy, precision, recall, f1, all_labels, all_preds)

#epoch_loss, accuracy, precision, recall, f1, all_labels, all_preds

def log_metrics(stage, stage_metrics, epoch):
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1']
    log_dict = {
        stage: {metric: value for metric, value in zip(metrics, stage_metrics)},
        'epoch': epoch
    }
    wandb.log(log_dict, step=epoch)
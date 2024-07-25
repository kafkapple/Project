from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

import numpy as np
import torch
from tqdm import tqdm
import wandb
from visualization import plot_confusion_matrix, save_and_log_figure
    
import os
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from tqdm import tqdm
import wandb
from hyperparameter_search import run_hyperparameter_sweep, run_sweep
from train_utils import evaluate_baseline
from data_utils import prep_data_for_benchmark

def compare_models(deep_model, train_loader, val_loader, test_loader, config, device):

    print(f'\nCalculating baseline model: SVM Classifier & Multi-class logistic Regression\n-max_iter: {config.max_iter}')
    
    # X_train, y_train = extract_features_and_labels(train_loader)
    # X_val, y_val = extract_features_and_labels(val_loader)
    # print(f"Shape of X_train: {X_train.shape}")

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_val_scaled = scaler.transform(X_val)
    X_train_scaled, y_train = prep_data_for_benchmark(train_loader)
    X_test_scaled, y_test = prep_data_for_benchmark(test_loader)
    # SVM 모델
    svm_path = os.path.join(config.MODEL_BASE_DIR, 'baseline_model_SVM.joblib')
    print(svm_path)
    config.model_benchmark = 'svm'
    new_baseline = False
    config.sweep_config = {
        'method': 'bayes',
        'metric': {'goal': 'minimize', 'name': 'val.loss' },
        'parameters': {
            'max_iter' : {'values' : [100, 1000, 10000]},
            'C_val': {'values': [0.1, 1, 10, 100, 1000]},
            'kernel': {'values': ['linear', 'rbf']},
            'gamma': {'values':[0.001, 0.0001]},
        }
        }
    config.CONFIG_DEFAULTS = {
        "resume":False,
        "architecture": "SVM",
        "dataset": f"{config.DATA_NAME}",
        "max_iter":1000,
        "C_val": 1,
        "kernel": 'rbf',
        'gamma': 0.001
        }  

    if os.path.exists(svm_path):
        user_input = input(f"SVM model found at {svm_path}. Load it? (y/n): ").lower()
        if user_input == 'y':
            svm_model = joblib.load(svm_path)
            X_test_scaled, y_test = prep_data_for_benchmark(test_loader)
            svm_loss, svm_accuracy, svm_precision, svm_recall, svm_f1, y_pred_svm = evaluate_baseline(svm_model, X_test_scaled, y_test, config)
            new_baseline = False
        else:
            
            new_baseline=True
    else:
        new_baseline=True
        
    if new_baseline:
        config.IS_SWEEP=True
        # num_epoch_sweep = input("Number of epoch / sweep for Hyperparameter Search: (e.g., 10 5 # 10 epoch / 5 sweep)")
        # config.NUM_EPOCHS, config.N_SWEEP = [int(i) for i in num_epoch_sweep.split(' ')]
        print(f'{config.NUM_EPOCHS}-epoch / {config.N_SWEEP}-sweep\n')
        print('New SVM model will be trained')
        svm_model = SVC(kernel=config.kernel, max_iter=config.max_iter, C = config.C_val)
        #run_sweep(config, train_loader, val_loader, svm_model)
        #joblib.dump(svm_model, svm_path)
        #print(svm_path)
        svm_model.fit(X_train_scaled, y_train)
        svm_loss, svm_accuracy, svm_precision, svm_recall, svm_f1, y_pred_svm = evaluate_baseline(svm_model, X_test_scaled, y_test, config)

    config.model_benchmark = 'LogisticRegression'
    config.sweep_config = {
        'method': 'bayes',
        'metric': {'goal': 'minimize', 'name': 'val.loss' },
        'parameters': {
            'max_iter' : {'values' : [100, 1000, 10000]},
            'C_val': {'values': [0.1, 1, 10, 100, 1000]},
            'solver': {'values': ['sage', 'sag', 'liblinear','lbfgs', 'newton-cg']},
            "penalty":{"values":['l1', 'l2','elasticnet']}
        }
        }
    config.CONFIG_DEFAULTS = {
            "resume":False,
            "architecture": 'LogisticRegression',
            "dataset": f"{config.DATA_NAME}",
            "max_iter":1000,
            "C_val": 1,
            "solver": 'saga',
            'penalty': 'l2'
            }  
    
    
    
    # Logistic Regression 모델      
    lr_path = os.path.join(config.MODEL_BASE_DIR, 'baseline_model_LogisticRegression.joblib')
    print(lr_path)
    if os.path.exists(lr_path):
        user_input = input(f"Logistic Regression model found at {lr_path}. Load it? (y/n): ").lower()
        if user_input == 'y':
            lr_model = joblib.load(lr_path)
            X_test_scaled, y_test = prep_data_for_benchmark(test_loader)
            lr_loss, lr_accuracy, lr_precision, lr_recall, lr_f1, y_pred_lr = evaluate_baseline(lr_model, X_test_scaled, y_test, config)
        else:
            new_baseline=True
    else:
        new_baseline=True
    if new_baseline:
        config.IS_SWEEP=True
        # num_epoch_sweep = input("Number of epoch / sweep for Hyperparameter Search: (e.g., 10 5 # 10 epoch / 5 sweep)")
        # config.NUM_EPOCHS, config.N_SWEEP = [int(i) for i in num_epoch_sweep.split(' ')]
        print(f'{config.NUM_EPOCHS}-epoch / {config.N_SWEEP}-sweep\n')
        
        print('New LR model will be trained')
   
        lr_model = LogisticRegression(multi_class='ovr', C = config.C_val, penalty = config.penalty, solver = config.solver, max_iter=config.max_iter, class_weight='balanced')
        lr_model.fit(X_train_scaled, y_train)
        lr_loss, lr_accuracy, lr_precision, lr_recall, lr_f1, y_pred_lr = evaluate_baseline(lr_model, X_test_scaled, y_test, config)
        #run_sweep(config, train_loader, val_loader, lr_model)
        # lr_accuracy, lr_precision, lr_recall, lr_f1 = train_and_evaluate_baseline(
            #     lr_model, X_train_scaled, y_train, X_val_scaled, y_val, config
            # )
        #joblib.dump(lr_model, lr_path)

    ### DL model
    deep_model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        #for features, labels in tqdm(test_loader, desc="Evaluating DL Model"):
        for batch in tqdm(test_loader, desc="Evaluating DL Model"):
            features = batch['audio'].to(device)
            labels = batch['label'].to(device)

            outputs = deep_model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    average = config.METRIC_AVG
    deep_accuracy = accuracy_score(y_true, y_pred)
    deep_precision = precision_score(y_true, y_pred, average=average)
    deep_recall = recall_score(y_true, y_pred, average=average)
    deep_f1 = f1_score(y_true, y_pred, average=average)
    Model_name = 'wav2vec'
    print(f"Deep Learning Model - {Model_name}:")
    print(f"Accuracy: {deep_accuracy:.4f}, Precision: {deep_precision:.4f}, Recall: {deep_recall:.4f}, F1: {deep_f1:.4f}")
    print("\nSVM Model:")   
    print(f"Accuracy: {svm_accuracy:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, F1: {svm_f1:.4f}")
    print("\nLogistic Regression Model:")
    print(f"Accuracy: {lr_accuracy:.4f}, Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1: {lr_f1:.4f}")

    wandb.log({
        f"{Model_name}": {
            "accuracy": deep_accuracy,
            "precision": deep_precision,
            "recall": deep_recall,
            "f1": deep_f1
        },
        "svm": {
            "accuracy": svm_accuracy,
            "precision": svm_precision,
            "recall": svm_recall,
            "f1": svm_f1
        },
        "logistic_regression": {
            "accuracy": lr_accuracy,
            "precision": lr_precision,
            "recall": lr_recall,
            "f1": lr_f1
        }
    })
    fig = plot_confusion_matrix(y_true, y_pred, config.LABELS_EMOTION)
    fig_lr = plot_confusion_matrix(y_true, y_pred_lr, config.LABELS_EMOTION)
    fig_svm = plot_confusion_matrix(y_true, y_pred_svm, config.LABELS_EMOTION)
    save_and_log_figure('test', fig, config, f'Confusion Matrix_{Model_name}', f'Confusion Matrix_{Model_name}')
    save_and_log_figure('test', fig_svm, config, 'Confusion Matrix_SVM', 'Confusion Matrix_SVM')
    save_and_log_figure('test', fig_lr, config, 'Confusion Matrix_LR', 'Confusion Matrix_LR')



# def train_and_evaluate_baseline(model, X_train, y_train, X_test, y_test, config):
#     model.fit(X_train, y_train)
#     return evaluate_baseline(model, X_test, y_test, config)
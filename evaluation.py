from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from tqdm import tqdm
import wandb
from visualization import plot_confusion_matrix

def extract_features_and_labels(dataloader):
    all_features = []
    all_labels = []
    for features, labels in tqdm(dataloader, desc="Extracting features"):
      # 만약 features가 3D (batch, sequence_length, feature_dim)라면 2D로 변환
      if features.dim() == 3:
          features = features.view(features.size(0), -1)
      all_features.append(features.cpu().numpy())
      all_labels.append(labels.cpu().numpy())
      return np.vstack(all_features), np.concatenate(all_labels)

def train_and_evaluate_baseline(model, X_train, y_train, X_test, y_test, config):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    plot_confusion_matrix(y_test, y_pred, config.LABELS_EMOTION)
    return accuracy, precision, recall, f1

def compare_models(deep_model, train_loader, val_loader, test_loader, config, device):
    deep_model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Evaluating DL Model"):
            features, labels = features.to(device), labels.to(device)
            outputs = deep_model(features)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    average='weighted' #'macro'
    deep_accuracy = accuracy_score(y_true, y_pred)
    deep_precision = precision_score(y_true, y_pred, average=average)
    deep_recall = recall_score(y_true, y_pred, average=average)
    deep_f1 = f1_score(y_true, y_pred, average=average)

    print('\nCalculating baseline model...')
    X_train, y_train = extract_features_and_labels(train_loader)
    #X_val, y_val = extract_features_and_labels(val_loader)
    X_test, y_test = extract_features_and_labels(test_loader)
    print(f"Shape of X_train: {X_train.shape}")  # 디버깅을 위해 shape 출력

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    #X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVC(kernel='rbf')
    svm_accuracy, svm_precision, svm_recall, svm_f1 = train_and_evaluate_baseline(
        svm_model, X_train_scaled, y_train, X_test_scaled, y_test, config
    )
    lr_model = LogisticRegression(multi_class='ovr', max_iter=1000)
    lr_accuracy, lr_precision, lr_recall, lr_f1 = train_and_evaluate_baseline(
        lr_model, X_train_scaled, y_train, X_test_scaled, y_test, config
    )

    print("Deep Learning Model:")
    print(f"Accuracy: {deep_accuracy:.4f}, Precision: {deep_precision:.4f}, Recall: {deep_recall:.4f}, F1: {deep_f1:.4f}")
    print("\nSVM Model:")
    print(f"Accuracy: {svm_accuracy:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, F1: {svm_f1:.4f}")
    print("\nLogistic Regression Model:")
    print(f"Accuracy: {lr_accuracy:.4f}, Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1: {lr_f1:.4f}")

    wandb.log({
        "deep_learning": {
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
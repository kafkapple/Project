from .models import list_models, unfreeze_layers, print_model_info, chk_best_model_info, find_best_model, prep_model, get_model, EmotionRecognitionModel_v2, EmotionRecognitionWithWav2Vec #SVMClassifier, 

#EmotionRecognitionModel_v1, 
# def get_model(config, train_loader):
#     input_size = train_loader.dataset[0][0].shape[1]
#     num_classes = len(config.LABELS_EMOTION)
    
#     if config.MODEL == "wav2vec_v1":
#         return EmotionRecognitionModel_v1(input_size, num_classes, config.DROPOUT_RATE, config.ACTIVATION)
#     elif config.MODEL == "wav2vec_v2":
#         return EmotionRecognitionModel_v2(input_size, num_classes, config.DROPOUT_RATE, config.ACTIVATION)
#     elif config.MODEL == "SVM_C":
#         return SVMClassifier(input_size, num_classes)
#     else:
#         raise ValueError(f"Unknown model type: {config.MODEL}")
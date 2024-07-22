import wandb
from models import prep_model#, get_model
from train_utils import train_model, log_metrics, evaluate_baseline
#from data_utils import prepare_dataloaders
from visualization import visualize_results#, plot_confusion_matrix
from data_utils import prep_data_for_benchmark


def run_hyperparameter_sweep(config, train_loader, val_loader, model=None):
    #config.WANDB_PROJECT= "NMA_Project_SER_sweep_test_v0"#"NMA_Project_SER_sweep_together_v1_e5"
    #print(config)
    EvaluationResult = config.EvaluationResult
    if config.CUR_MODE == "benchmark":
        X_train_scaled, y_train = prep_data_for_benchmark(train_loader)
        X_val_scaled, y_val = prep_data_for_benchmark(val_loader)
        if config.model_benchmark == 'LogisticRegression':
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
                    'penalty': 'elasticnet'

                    }  
            
        elif config.model_benchmark =='svm':
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
            
        else:
            print('Error. no benchmark model')
            
    if config.CUR_MODE =='benchmark':
        config.WANDB_PROJECT="Benchmark"
        sweep_id = wandb.sweep(config.sweep_config, project=config.WANDB_PROJECT)
        
    else: #config.SWEEP_NAIVE:
        sweep_id="8lad6k0u" #
        config.WANDB_PROJECT="NMA_Project_SER_sweep_together_v1_e5"
        
        sweep_id = f"{config.ENTITY}/{config.WANDB_PROJECT}/{sweep_id}"
        
        
        sweep_id_full = f"{config.ENTITY}/{config.WANDB_PROJECT}/{sweep_id}"#wandb.sweep
        config.sweep_id = sweep_id_full
        print(f'\nSweep starts. Sweep id: {sweep_id}\n')

        # sweep_id = config.sweep_id
        # print(f'Previous Sweep Sweep id is loaded : {sweep_id}')

    def train():
        
        # id_wandb=wandb.util.generate_id()
        # #if not config.CUR_MODE == "benchmark":
        # wandb.init(
        # id=id_wandb,
        # project=config.WANDB_PROJECT,
        
        #     #name=config.WANDB_NAME, #
        #     config=config.CONFIG_DEFAULTS,
        #     resume=False
        # )
        # else: # benchmark
            
        print('after init')
        if config.CUR_MODE == "benchmark":
            config.max_iter = wandb.config.max_iter
            config.C_val = wandb.config.C_val
            if config.model_benchmark == 'LogisticRegression':
                config.solver = wandb.config.solver
                config.penalty = wandb.config.penalty
            elif config.model_benchmark =='svm':
                config.kernel = wandb.config.kernel
                config.gamma = wandb.config.gamma
            else:
                print('err')
            model.fit(X_train_scaled, y_train)
            loss, accuracy, precision, recall, f1, y_pred = evaluate_baseline(model, X_val_scaled, y_val, config)
            log_metrics('val', EvaluationResult(loss, accuracy, precision, recall, f1, y_val, y_pred))
            print('\nafter log_metric\n')
        else:
            # Update config with sweep parameters
            config.lr = wandb.config.lr
            config.BATCH_SIZE = wandb.config.batch_size
            config.DROPOUT_RATE = wandb.config.dropout_rate
            config.ACTIVATION = wandb.config.activation
            
            #train_loader, val_loader, _ = prepare_dataloaders(data, labels, config)
            #model = get_model(config, train_loader)
            model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=True)
            #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            train_model(model, train_loader, val_loader, config, device, optimizer, criterion)
            visualize_results(config, model, val_loader, device, config.history, 'test')
    
    wandb.agent(sweep_id, function=train, count=config.N_SWEEP)
    wandb.finish()
 
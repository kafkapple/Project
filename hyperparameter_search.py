import wandb
from models import prep_model, get_model
from train_utils import train_model
from data_utils import prepare_dataloaders

def run_hyperparameter_sweep(config, data, labels):
    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.1},
            'batch_size': {'values': [16, 32, 64, 128]},
            'num_epochs': {'min': 5, 'max': 50},
            'dropout_rate': {'min': 0.1, 'max': 0.5},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=config.WANDB_PROJECT)

    def train():
        wandb.init()
        
        # Update config with sweep parameters
        config.LEARNING_RATE = wandb.config.learning_rate
        config.BATCH_SIZE = wandb.config.batch_size
        config.NUM_EPOCHS = wandb.config.num_epochs
        config.DROPOUT_RATE = wandb.config.dropout_rate
        train_loader, val_loader, _ = prepare_dataloaders(data, labels, config)
        #model = get_model(config, train_loader)
        model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=True)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_model(model, train_loader, val_loader, config, device, optimizer, criterion)

    wandb.agent(sweep_id, train, count=config.N_SWEEP)
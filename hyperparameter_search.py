import wandb
from models import prep_model, get_model
from train_utils import train_model
from data_utils import prepare_dataloaders

def run_hyperparameter_sweep(config, data, labels):

    config.WANDB_PROJECT= "NMA_Project_SER_sweep_test_v0"#"NMA_Project_SER_sweep_together_v1_e5"
    #print(config)
    if config.SWEEP_NAIVE:
        sweep_id = wandb.sweep(config.sweep_config, project=config.WANDB_PROJECT)
        sweep_id = f"{config.ENTITY}/{config.WANDB_PROJECT}/{sweep_id}"#wandb.sweep
        config.sweep_id = sweep_id
        print(f'\nFirst Sweep starts. Sweep id: {sweep_id}\n')
    else:
        sweep_id = config.sweep_id
        print(f'Previous Sweep Sweep id is loaded : {sweep_id}')
    print(type(config))

    def train():
        wandb.init()
        
        # Update config with sweep parameters
        config.LEARNING_RATE = wandb.config.learning_rate
        config.BATCH_SIZE = wandb.config.batch_size
        config.DROPOUT_RATE = wandb.config.dropout_rate
        config.ACTIVATION = wandb.config.activation
        train_loader, val_loader, _ = prepare_dataloaders(data, labels, config)
        #model = get_model(config, train_loader)
        model, optimizer, criterion, device = prep_model(config, train_loader, is_sweep=True)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        train_model(model, train_loader, val_loader, config, device, optimizer, criterion)

    wandb.agent(sweep_id, train, count=config.N_SWEEP)
    wandb.finish()
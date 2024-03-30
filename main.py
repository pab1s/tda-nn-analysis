import torch
import yaml
from datetime import datetime
from utils.data_utils import get_dataloaders
from models import get_model
from trainers import get_trainer
from os import path

def main(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(config)
    
    model = get_model(
        config['model']['name'], 
        config['model']['num_classes'], 
        pretrained=config['model']['pretrained']
    ).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    optimizer_params = {'lr': config['training']['learning_rate']}
    
    # Prepare filenames for logging and plotting
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dataset_time = f"{config['model']['name']}_{config['data']['name']}_{current_time}"
    log_filename = path.join(config['paths']['log_path'], f"log_{model_dataset_time}.csv")
    plot_filename = path.join(config['paths']['plot_path'], f"plot_{model_dataset_time}.png")
    
    trainer = get_trainer(config['trainer'], model=model, device=device)
    
    trainer.build(
        criterion=criterion, 
        optimizer_class=optimizer, 
        optimizer_params=optimizer_params
    )
    trainer.train(
        train_loader=train_loader, 
        num_epochs=config['training']['num_epochs'], 
        log_path=log_filename, 
        plot_path=plot_filename
    )
    trainer.evaluate(test_loader=test_loader)

if __name__ == "__main__":
    main("config/config.yaml")

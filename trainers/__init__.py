from trainers.basic_trainer import BasicTrainer

def get_trainer(trainer_name, **kwargs):
    if trainer_name == "BasicTrainer":
        return BasicTrainer(**kwargs)
    else:
        raise ValueError(f"Trainer {trainer_name} not recognized.")

from trainers.basic_trainer import BasicTrainer

def get_trainer(trainer_name, **kwargs):
    """
    Returns an instance of the specified trainer.

    Parameters:
    - trainer_name (str): The name of the trainer.
    - **kwargs: Additional keyword arguments to be passed to the trainer constructor.

    Returns:
    - Trainer: An instance of the specified trainer.

    Raises:
    - ValueError: If the trainer name is not recognized.
    """
    if trainer_name == "BasicTrainer":
        return BasicTrainer(**kwargs)
    else:
        raise ValueError(f"Trainer {trainer_name} not recognized.")

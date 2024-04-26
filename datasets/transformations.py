from torchvision import transforms as T

def get_transforms(transform_configs):
    """
    Returns a composed transformation object based on the provided list of configurations.

    Args:
        transform_configs (list of dict): A list containing dictionaries that specify each transformation.

    Returns:
        torchvision.transforms.Compose: A composed transformation object.

    Raises:
        ValueError: If the specified transform is not recognized or parameters are missing.

    Example:
        transform_configs = [
            {'type': 'Resize', 'parameters': {'size': [240, 240]}},
            {'type': 'ToTensor'},
            {'type': 'Normalize', 'parameters': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}
        ]
        transforms = get_transforms(transform_configs)
    """
    transform_list = []
    for transform_config in transform_configs:
        transform_type = transform_config['type']
        parameters = transform_config.get('parameters', {})

        if hasattr(T, transform_type):
            transform_class = getattr(T, transform_type)
            transform = transform_class(**parameters) if parameters else transform_class()
            transform_list.append(transform)
        else:
            raise ValueError(f"Transform {transform_type} not recognized.")

    return T.Compose(transform_list)

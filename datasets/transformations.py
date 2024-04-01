from torchvision import transforms

def get_transforms(config):
    transform_list = []
    for transform_config in config['data']['transforms']:
        transform_name = transform_config['name']
        parameters = transform_config.get('parameters', {})

        if hasattr(transforms, transform_name):
            transform_class = getattr(transforms, transform_name)
            transform = transform_class(**parameters)
            transform_list.append(transform)
        else:
            raise ValueError(f"Transform {transform_name} not recognized.")

    composed_transform = transforms.Compose(transform_list)
    return composed_transform
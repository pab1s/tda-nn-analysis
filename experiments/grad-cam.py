import torch
import yaml
import argparse
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_pil_image
from datasets.dataset import get_dataset
from datasets.transformations import get_transforms
from factories.model_factory import ModelFactory
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def modify_state_dict_keys(state_dict):
    """
    Adjust the keys in the loaded state dictionary to match 
    the expected keys of the model.
    """
    
    new_state_dict = {}
    for key, value in state_dict.items():
        if "classifier.0" in key or "classifier.3" in key:
            new_key = key.replace("classifier.", "classifier.1.")
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def get_target_layers(model, model_type):
    """
    Returns a list of suitable convolutional layers for Grad-CAM.

    Args:
        model (torch.nn.Module): The model to extract target layers from.
        model_type (str): The type of the model.

    Returns:
        list: A list of convolutional layers for Grad-CAM.
    
    Raises:
        ValueError: If the model type is not supported.
    """

    if model_type == 'efficientnet_b0':
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and 'block' in name:
                layers.append(name)
    elif model_type == 'densenet121':
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and 'denseblock' in name:
                layers.append(name)
    else:
        raise ValueError(f"Unsupported model type for Grad-CAM: {model_type}")
    
    return layers

def evaluate_model_with_grad_cam(config_path, model_path):
    """
    Evaluate a trained model with Grad-CAM.

    Args:
        config_path (str): The path to the configuration file.
        model_path (str): The path to the trained model file (.pth).
    """

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        exit()
    device = torch.device("cuda")

    transforms = get_transforms(config['data']['transforms'])
    eval_transforms = get_transforms(config['data']['eval_transforms'])
    data = get_dataset(config['data']['name'], config['data']['dataset_path'], train=True, transform=transforms)

    test_size = int(len(data) * config['data']['test_size'])
    val_size = int(len(data) * config['data']['val_size'])
    train_size = len(data) - test_size - val_size
    _, data_test = random_split(data, [train_size + val_size, test_size], generator=torch.Generator().manual_seed(config['random_seed']))

    data_test.dataset.transform = eval_transforms
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False)

    model_factory = ModelFactory()
    model = model_factory.create(config['model']['type'], num_classes=config['model']['parameters']['num_classes'], pretrained=config['model']['parameters']['pretrained']).to(device)
    for name, param in model.named_parameters():
        print(name)
    loaded_state_dict = torch.load(model_path)
    corrected_state_dict = modify_state_dict_keys(loaded_state_dict)
    model.load_state_dict(corrected_state_dict)
    model.eval()

    target_layers = get_target_layers(model, config['model']['type'])
    cam_extractor = LayerCAM(model, target_layer=target_layers)
    model_name = os.path.basename(model_path).split('.')[0]
    output_dir = os.path.join('grad-cam', model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (transformed_image, targets) in enumerate(test_loader):
        input_tensor = transformed_image.to(device)
        out = model(input_tensor)
        predicted_class = out.argmax(dim=1).item()
        correct = 'true' if predicted_class == targets.item() else 'false'

        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
        fused_cam = cam_extractor.fuse_cams(cams)
        fused_cam = fused_cam.squeeze(0)

        # Retrieve original image using new method
        original_image = data.get_original_image(idx)
        pil_fused_cam = to_pil_image(fused_cam, mode='F')
        fused_cam_image = overlay_mask(original_image, pil_fused_cam, alpha=0.5)

        plt.imshow(fused_cam_image)
        plt.axis('off')
        plt.title(f"Predicted: {predicted_class} ({correct})")
        plt.show()

        # Convert the overlay image to an array and scale it for saving
        fused_cam_array = np.array(fused_cam_image)
        fused_cam_scaled = np.uint8(fused_cam_array * 255)
        fused_pil_image = Image.fromarray(fused_cam_scaled)
        fused_pil_image.save(os.path.join(output_dir, f"{idx}_{correct}.png"))

        # Remove hooks to avoid memory leaks
        cam_extractor.remove_hooks()

    print(f"Grad-CAMs saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained model with Grad-CAM.')
    parser.add_argument('config_filename', type=str, help='Filename of the configuration file within the "config" directory')
    parser.add_argument('model_path', type=str, help='Path to the trained model file (.pth)')
    args = parser.parse_args()

    config_path = f"config/{args.config_filename}"
    evaluate_model_with_grad_cam(config_path, args.model_path)

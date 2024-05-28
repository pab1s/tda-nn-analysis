import os
from PIL import Image
from torch.utils.data import Dataset

class CarDataset(Dataset):
    def __init__(self, data_dir, train=None, transform=None):
        """
        CarDataset class represents a dataset of car images.

        Args:
            data_dir (string): Path to the dataset directory.
            train (bool, optional): Whether the dataset is for training or not. Default is None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels, self.idx_to_class = self._load_dataset()

    def _load_dataset(self):
        """
        Load the dataset from the data directory.

        Returns:
            images (list): List of image paths.
            labels (list): List of corresponding labels.
            idx_to_class (dict): Mapping of label index to class name.
        """
        images = []
        labels = []
        label_to_idx = {}
        idx_to_class = {}
        current_label = 0

        # Walk through the data directory
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    label_name = os.path.basename(root)
                    if label_name not in label_to_idx:
                        label_to_idx[label_name] = current_label
                        idx_to_class[current_label] = label_name
                        current_label += 1
                    label = label_to_idx[label_name]

                    images.append(os.path.join(root, filename))
                    labels.append(label)

        return images, labels, idx_to_class

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self):
        """
        Return the list of class names.

        Returns:
            list: List of class names.
        """
        return [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]

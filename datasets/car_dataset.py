import os
from PIL import Image
from torch.utils.data import Dataset

class CarDataset(Dataset):
    def __init__(self, data_dir, train=None, transform=None):
        """
        Args:
            data_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images, self.labels, self.idx_to_class = self._load_dataset()

    def _load_dataset(self):
        images = []
        labels = []
        label_to_idx = {}
        idx_to_class = {}
        current_label = 0

        # Walk through the data directory
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith(".jpg"):
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
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_classes(self):
        """Return the list of class names."""
        return [self.idx_to_class[idx] for idx in sorted(self.idx_to_class)]

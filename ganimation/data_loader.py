import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, transform, mode, c_dim):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.transform = transform
        self.mode = mode
        self.c_dim = c_dim

        self.train_dataset = []
        self.test_dataset = []

        # Fills train_dataset and test_dataset --> [filename, boolean attribute vector]
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

        print("------------------------------------------------")
        print("Training images: ", len(self.train_dataset))
        print("Testing images: ", len(self.test_dataset))

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        lines = lines[1:]
        random.shuffle(lines)

        # Extract the info from each line
        for idx, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]
            # Vector representing the presence of each attribute in each image
            label = []  

            for n in range(self.c_dim):
                label.append(float(values[n])/5.)

            if idx < 100:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Dataset ready!...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, attr_path, c_dim=17,
               batch_size=16, mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, transform, mode, c_dim)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader


if __name__ == "__main__":
    from types import SimpleNamespace
    
    config = SimpleNamespace(**{
        "image_dir": "ganimation/data/celeba/images_aligned",
        "attr_path": "ganimation/data/celeba/list_attr_celeba.txt",
        "batch_size": 32
    })
    dl = get_loader(config.image_dir, config.attr_path, batch_size=32)
    
    for x, c in dl:
        print(x.shape)
        print(c.shape)
        break
    print("Done!")
import logging
import os

from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextDataset(Dataset):
    def __init__(self, data_folder: str, transform):
        self.transform = transform
        IMG_EXTENSIONS = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'}

        # Sorting the elements in the data based on batch length
        self.image_paths = []

        self.root_path = os.path.abspath(data_folder)

        for f in os.listdir(data_folder):
            f_path = os.path.join(data_folder, f)

            if os.path.isdir(f_path):
                for f2 in os.listdir(f_path):
                    extension = f2[f2.rfind("."):]
                    if extension in IMG_EXTENSIONS:
                        self.image_paths.append(f + "/" + f2)
            else:
                extension = f[f.rfind("."):]
                if extension in IMG_EXTENSIONS:
                    self.image_paths.append(f)

        print("Loaded image paths", self.root_path, len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item: int):
        path = os.path.join(self.root_path, self.image_paths[item])

        # Make sure not to deal with rgba or grayscale images.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"images": image, "paths": self.image_paths[item]}

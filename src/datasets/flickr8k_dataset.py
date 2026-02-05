import os
from PIL import Image
from torch.utils.data import Dataset

class Flickr8kDataset(Dataset):
   
    def __init__(self, root: str, split: str = "train", transform=None):
        self.root = root
        self.transform = transform

        img_dir = os.path.join(root, "images")
        cap_dir = os.path.join(root, "captions")

        #legge gli split file
        split_map = {
            "train": "Flickr_8k.trainImages.txt",
            "val": "Flickr_8k.devImages.txt",
            "test": "Flickr_8k.testImages.txt",
        }
        if split not in split_map:
            raise ValueError("split must be train/val/test")

        #legge le caption
        split_file = os.path.join(cap_dir, split_map[split])
        token_file = os.path.join(cap_dir, "Flickr8k.token.txt")

        # lista immagini nello split
        with open(split_file, "r", encoding="utf-8") as f:
            self.images = [line.strip() for line in f if line.strip()]

        # caption 
        self.captions = {}
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" not in line:
                    continue
                img_id, caption = line.strip().split("\t", 1)
                img_name = img_id.split("#")[0]
                self.captions.setdefault(img_name, []).append(caption)

        self.img_dir = img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        path = os.path.join(self.img_dir, img_name)
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        captions = self.captions.get(img_name, [])
        """
        Ritorna:
            - image 
            - captions 
            - image_id 
        """
        return {"image": image, "captions": captions, "image_id": img_name}

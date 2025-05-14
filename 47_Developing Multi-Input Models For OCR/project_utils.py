import torch
from torch.utils.data import Dataset
import random, string
from PIL import ImageFont, Image, ImageDraw
from torchvision.transforms import ToTensor

class ProjectDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.data = []
        self.labels = []
        self.label_mapping = {'primary_id': 0, 'secondary_id': 1}  
        self.type_mapping = {'home': 0, 'life': 1, 'auto': 2, 'health': 3, 'other':4}  
        self.generate_data()

    def generate_data(self):
        for _ in range(self.num_samples):
            text_type = random.choice(['home', 'life', 'auto', 'health', 'other'])
            text_type_label = random.choice(['primary_id', 'secondary_id'])
            text = self.generate_random_string()
            image = self.text_to_image(text)
            type_vector = self.text_type_to_vector(text_type)
            label_index = self.label_mapping[text_type_label]  
            self.data.append((ToTensor()(image), type_vector))
            self.labels.append(label_index)  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def generate_random_string(length=5):
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))

    @staticmethod
    def text_type_to_vector(text_type):
        types = ['home', 'life', 'auto', 'health', 'other']
        vector = [0] * len(types)
        vector[types.index(text_type)] = 1
        return torch.tensor(vector, dtype=torch.float)

    @staticmethod
    def text_to_image(text):
        font = ImageFont.load_default()
        size = font.getbbox(text)[2:]
        image = Image.new('L', size, "white")
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, fill="black", font=font)
        return image.resize((64, 64))  
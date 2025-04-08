from ast import literal_eval
import pandas as pd
import numpy as np
import torch
import convertor
import torchvision

TITLE = 1
INGREDIENTS = 2
INSTRUCTIONS = 3
IMAGE_NAME = 4
CLEANED_INGREDIENTS = 5


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file="Food Ingredients and Recipe Dataset with Image Name Mapping.csv", transform=None):
        self.data = pd.read_csv(csv_file)
        
        self.data['Ingredients'] = self.data['Ingredients'].apply(literal_eval)
        self.data['Cleaned_Ingredients'] = self.data['Cleaned_Ingredients'].apply(literal_eval)
        self.data = self.data.to_numpy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        template = np.random.choice(convertor.TEMPLATES)
        title = self.data[idx][TITLE]
        ingredients = self.data[idx][CLEANED_INGREDIENTS]
        instructions = self.data[idx][INSTRUCTIONS]
        text = template(title, ingredients, [instructions])
        images = convertor.convert_to_image(text, format='markdown')
        
        return torchvision.transforms.functional.to_tensor(np.hstack(images)),\
            title, ingredients, instructions


def main():
    r = RecipeDataset()
    for data in r:
        img, title, ingredients, instructions = data
        convertor.show_image(img.permute(1, 2, 0).numpy())


if __name__ == "__main__":
    main()
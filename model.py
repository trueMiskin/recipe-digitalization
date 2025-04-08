import argparse
import npfl138
from npfl138 import trainable_module as tm
from npfl138 import global_keras_initializers
import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
from dataset import RecipeDataset
from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description='Recipe digitalization')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--threads', type=int, default=8, help='Number of threads')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in dataloader')


class Model(tm.TrainableModule):
    def __init__(self):
        super().__init__()
        # Model definition
        # self.conv1 = torch.nn.Conv2d(1, 16, 3, stride=2)
        # self.conv2 = torch.nn.Conv2d(16, 32, 3, padding="same")

    def forward(self, x):
        # Model calling
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # return x
        pass

def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Set number of threads if > 0; otherwise, use as many threads as cores.
    if args.threads is not None and args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    global_keras_initializers()

    model = Model()
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.002),
        loss=torch.nn.MSELoss()
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def transform(images, title, ingredients, instructions):
        nonlocal tokenizer
        # TODO: img augmentation
        title = tokenizer(title, return_tensors='pt')['input_ids'].squeeze()
        ingredients = tokenizer("[SEP]".join(ingredients), return_tensors='pt')['input_ids'].squeeze()
        instructions = tokenizer("[SEP]".join(instructions), return_tensors='pt')['input_ids'].squeeze()
        return images, (title, ingredients, instructions)

    train_dataset, test_dataset = \
        torch.utils.data.random_split(
            RecipeDataset(),
            [0.9, 0.1],
            torch.Generator().manual_seed(1)
    )
    
    transformed_train = npfl138.TransformedDataset(train_dataset)
    transformed_test = npfl138.TransformedDataset(test_dataset)
    transformed_train.transform = transformed_test.transform = transform

    def collate_fn(batch):
        images, targets = zip(*batch)
        title, ingredients, instructions = zip(*targets)
        images = torch.stack(images)
        title = torch.nn.utils.rnn.pack_sequence(title, enforce_sorted=False)
        ingredients = torch.nn.utils.rnn.pack_sequence(ingredients, enforce_sorted=False)
        instructions = torch.nn.utils.rnn.pack_sequence(instructions, enforce_sorted=False)
        return images, (title, ingredients, instructions)
    transformed_train.collate = transformed_test.collate = collate_fn

    train = transformed_train.dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test = transformed_test.dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.fit(train, epochs=5)

if __name__ == '__main__':
    main(parser.parse_args())
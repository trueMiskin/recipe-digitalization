import os
import datetime
import re
import argparse
import npfl138
from npfl138 import trainable_module as tm
from npfl138 import global_keras_initializers
import torch
import torch.nn.functional as F
import numpy as np
import random
from dataset import RecipeDataset, OnlyImageRecipeDataset
import torchmetrics
from transformers import AutoTokenizer, ElectraForSequenceClassification

parser = argparse.ArgumentParser(description='Recipe digitalization')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--threads', type=int, default=1, help='Number of threads')
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in dataloader')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--random_contex_cutoff', type=float, default=0.5, help='Random context cutoff')
parser.add_argument('--show_prediction', default=False, action='store_true', help='Show predicted function')
parser.add_argument('--model', default=None, help="Load model")
parser.add_argument('--img_dir', default=None, help="Image directory")

class RandomizeContextCutoff(torch.nn.Module):
    def __init__(self, cutoff=0.5):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, x):
        if self.training:
            mask = torch.rand((x.shape[0]), device=x.device) < self.cutoff
            cutted_x = torch.arange(x.shape[1], device=x.device) < 15 * x
            x = torch.where(mask.unsqueeze(-1),
                        cutted_x,
                        x
            )
        return x

class Model(tm.TrainableModule):
    def __init__(self, backbone, random_contex_cutoff):
        super().__init__()
        self.backbone = backbone
        self.randomized_cutoff = RandomizeContextCutoff(random_contex_cutoff)
        self.ocr = None

    def forward(self, title, ingredients, description):
        title, _ = torch.nn.utils.rnn.pad_packed_sequence(title, batch_first=True)
        title_output = self.backbone(title, attention_mask=self.randomized_cutoff(title != 0))
        ingredients, _ = torch.nn.utils.rnn.pad_packed_sequence(ingredients, batch_first=True)
        ingredients_output = self.backbone(ingredients, attention_mask=self.randomized_cutoff(ingredients != 0))
        description, _ = torch.nn.utils.rnn.pad_packed_sequence(description, batch_first=True)
        description_output = self.backbone(description, attention_mask=self.randomized_cutoff(description != 0))
        return title_output, ingredients_output, description_output

    def predict(self, image):
        if self.ocr is None:
            from paddleocr import PPStructure
            self.ocr = PPStructure(show_log=False,
                layout_score_threshold=0.3,
                layout_nms_threshold=0.5,
                table=False,
                image_orientation=False,
                ocr=True,
                lang='en',
                merge_no_span_structure=False,
            )
        ocr_result = self.ocr(image)
        model_input = []
        bboxes = []
        for layout in ocr_result:
            previous_lines = ""
            for res in layout['res']:
                text = res['text']
                previous_lines += text
                if text.endswith('-'):
                    # connect split word
                    previous_lines = previous_lines[:-1]
            model_input.append(previous_lines)
            bboxes.append(layout['bbox'])

        tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
        tokenized_input = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True)
        input_ids = tokenized_input['input_ids'].to(self.device)
        attention_mask = tokenized_input['attention_mask'].to(self.device)
        self.backbone.eval()
        with torch.no_grad():
            logits = self.backbone(input_ids, attention_mask=attention_mask).logits
            return model_input, bboxes, torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().numpy()

    def compute_metrics(self, y_pred, y, *xs):
        """Compute and return metrics given the inputs, predictions, and target outputs.

        Parameters:
          y_pred: The model predictions, either a single tensor or a sequence of tensors.
          y: The target output of the model, either a single tensor or a sequence of tensors.
          *xs: The inputs to the model, unpacked, if the input was a sequence of tensors.

        Returns:
          logs: A dictionary of computed metrics.
        """
        y_pred = torch.cat([y_pred[0].logits, y_pred[1].logits, y_pred[2].logits], dim=0)
        y = torch.cat([y[0], y[1], y[2]], dim=0)
        for metric in self.metrics.values():
            metric.update(y_pred, y)
        return {name: metric.compute() for name, metric in self.metrics.items()}


def loss(y_pred, y_true):
    title_pred, ingredients_pred, description_pred = y_pred
    title_true, ingredients_true, description_true = y_true
    title_loss = F.cross_entropy(title_pred.logits, title_true)
    ingredients_loss = F.cross_entropy(ingredients_pred.logits, ingredients_true)
    description_loss = F.cross_entropy(description_pred.logits, description_true)
    return title_loss + ingredients_loss + description_loss


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, module, epoch, logs):
        if self.early_stop(logs['dev_loss']):
            print(f"Early stopping at epoch {epoch + 1}")
            return tm.TrainableModule.STOP_TRAINING

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


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

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    backbone_model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    model = Model(backbone_model, args.random_contex_cutoff)
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        loss=loss,
        logdir=args.logdir,
        metrics={'acc': torchmetrics.Accuracy('multiclass', num_classes=3)},
    )

    def transform(title, ingredients, instructions):
        nonlocal tokenizer
        title = tokenizer(title, return_tensors='pt')['input_ids'].squeeze()
        ingredients = tokenizer(" ".join(ingredients), return_tensors='pt')['input_ids'].squeeze()
        ingredients = ingredients[:512]
        instructions = tokenizer(" ".join(instructions), return_tensors='pt')['input_ids'].squeeze()
        instructions = instructions[:512]
        return (title, ingredients, instructions), (torch.tensor(0), torch.tensor(1), torch.tensor(2))

    train_dataset, test_dataset = \
        torch.utils.data.random_split(
            RecipeDataset(),
            [0.9, 0.1],
            torch.Generator().manual_seed(1)
    )
    
    transformed_train = npfl138.TransformedDataset(train_dataset)
    transformed_dev = npfl138.TransformedDataset(test_dataset)
    transformed_train.transform = transformed_dev.transform = transform

    def collate_fn(batch):
        data, target = zip(*batch)
        title, ingredients, instructions = zip(*data)
        title_gold, ingredients_gold, instructions_gold = zip(*target)
        title = torch.nn.utils.rnn.pack_sequence(title, enforce_sorted=False)
        ingredients = torch.nn.utils.rnn.pack_sequence(ingredients, enforce_sorted=False)
        instructions = torch.nn.utils.rnn.pack_sequence(instructions, enforce_sorted=False)
        return (title, ingredients, instructions), (torch.stack(title_gold), torch.stack(ingredients_gold), torch.stack(instructions_gold))
    transformed_train.collate = transformed_dev.collate = collate_fn

    train = transformed_train.dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dev = transformed_dev.dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.show_prediction:
        assert(args.model != None)
        model.load_weights(args.model)
        
        dataset = RecipeDataset(generate_images=True)
        if args.img_dir is not None:
            dataset = OnlyImageRecipeDataset(args.img_dir)

        for data in dataset:
            img, *_ = data
            img *= 255
            img = img.type(torch.uint8)
            img = img.permute(1, 2, 0).numpy()
            model_input, bboxes, classes = model.predict(img)

            from paddleocr import draw_structure_result
            result_dict = []
            for input, bbox, class_ in zip(model_input, bboxes, classes):
                result_dict.append({
                    'type': ['title', 'ingredients', 'description'][class_],
                    'bbox': bbox,
                    'res': '',
                    'img_idx': 0,
                    'score': 0.99,
                })

            final_output = draw_structure_result(img, result_dict, font_path='simfang.ttf')
            import convertor
            convertor.show_image(final_output)
    else:
        model.fit(train, dev=dev, epochs=args.epochs, callbacks=[EarlyStopper(patience=3)])

        model.save_weights(os.path.join(args.logdir, "model_weights.pt"),
                        os.path.join(args.logdir, "optimizer"))

if __name__ == '__main__':
    main(parser.parse_args())

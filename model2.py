import os
import datetime
import re
import argparse
from npfl138 import global_keras_initializers
import torch
import numpy as np
import random
from dataset import PreprocessedRecipeDataset, OnlyImageRecipeDataset
from PIL import Image
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torchmetrics.text import BLEUScore
import Levenshtein

parser = argparse.ArgumentParser(description='Recipe digitalization')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--threads', type=int, default=1, help='Number of threads')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--show_prediction', default=False, action='store_true', help='Show predicted function')
parser.add_argument('--model', default=None, help="Load model")
parser.add_argument('--img_dir', default=None, help="Image directory")

#     def predict(self, image):
#         if self.ocr is None:
#             from paddleocr import PaddleOCR
#             self.ocr = PaddleOCR(lang='en',
#                                 use_angle_cls=True
#             )
#         model_input = []
#         bboxes = []
#         res_part = []
        
#         result = self.ocr.ocr(image, det=True, rec=True)[0]
#         boxes = [line[0] for line in result]
#         txts = [line[1][0] for line in result]
#         # use OCR regions
#         model_input = txts
#         bboxes = []
#         res_part = []
#         for box, txt in zip(boxes, txts):
#             left, upper = box[0][0], box[0][1]
#             right, lower = box[2][0], box[2][1]
#             bboxes.append([left, upper, right, lower])
#             res_part.append([{'text': txt, 'text_region': box, 'confidence': 0.99}])

#         tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator")
#         tokenized_input = tokenizer(model_input, return_tensors='pt', padding=True, truncation=True)
#         input_ids = tokenized_input['input_ids'].to(self.device)
#         attention_mask = tokenized_input['attention_mask'].to(self.device)
#         self.backbone.eval()
#         with torch.no_grad():
#             logits = self.backbone(input_ids, attention_mask=attention_mask).logits
#             return model_input, bboxes, res_part, torch.argmax(torch.softmax(logits, dim=-1), dim=-1).cpu().numpy()

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


    MODEL_NAME = "google/flan-t5-small"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIM = 3

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.logdir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=WEIGHT_DECAY,
        save_total_limit=SAVE_TOTAL_LIM,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        report_to="tensorboard"
    )

    train_dataset, test_dataset = \
        torch.utils.data.random_split(
            PreprocessedRecipeDataset(tokenizer),
            [0.9, 0.1],
            torch.Generator().manual_seed(1)
    )

    def compute_metrics(eval_pred):
        predictions, target_ans = eval_pred

        target_ans = np.where(target_ans != -100, target_ans, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(target_ans, skip_special_tokens=True)

        bleu_metric = BLEUScore()
        bleu_labels = [[label] for label in decoded_labels]
        bleu = bleu_metric(decoded_preds, bleu_labels, n_gram=3).item()

        # Edit distance ratio
        ratio_sum = 0.0
        levanstein_9 = 0.0
        levanstein_8 = 0.0
        for pred, label in zip(decoded_preds, decoded_labels):
            r = Levenshtein.ratio(pred, label)
            ratio_sum += r
            levanstein_9 += (1.0 if r > 0.9 else 0.0)
            levanstein_8 += (1.0 if r > 0.8 else 0.0)

        return {"bleu": bleu, "levenshtein_ratio": ratio_sum / len(decoded_preds),
                "levenshtein_above_0.9": levanstein_9 / len(decoded_preds),
                "levenshtein_above_0.8": levanstein_8 / len(decoded_preds)}

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # if args.show_prediction:
    #     assert(args.model != None)
    #     model.load_weights(args.model)
        
    #     dataset = RecipeDataset(generate_images=True)
    #     if args.img_dir is not None:
    #         dataset = OnlyImageRecipeDataset(args.img_dir)

    #     # create logdir if it doesn't exist
    #     if not os.path.exists(args.logdir):
    #         os.makedirs(args.logdir)

    #     for idx, data in enumerate(dataset):
    #         img, *_ = data
    #         model_input, bboxes, res_part, classes = model.predict(img)

    #         from paddleocr import draw_structure_result
    #         result_dict = []
    #         for input, bbox, rest_p, class_ in zip(model_input, bboxes, res_part, classes):
    #             print(input)
    #             result_dict.append({
    #                 'type': ['title', 'ingredients', 'description'][class_],
    #                 'bbox': bbox,
    #                 'res': rest_p,
    #                 'img_idx': 0,
    #                 'score': 0.99,
    #             })

    #         final_output = draw_structure_result(img, result_dict, font_path='simfang.ttf')
    #         import convertor
    #         convertor.show_image(final_output)
    #         Image.fromarray(final_output).save(os.path.join(args.logdir, f"{idx:02d}.png"))

if __name__ == '__main__':
    main(parser.parse_args())

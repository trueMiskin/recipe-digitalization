import os
import datetime
import re
import argparse
from npfl138 import global_keras_initializers
import torch
import numpy as np
import random
from dataset import PreprocessedRecipeDataset, OnlyImageRecipeDataset, RecipeDataset
from dataset import extract_data_from_ocr_result, merge_close_boxes, prepare_question, QUESTIONS
from PIL import Image
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torchmetrics.functional.text import bleu_score
import Levenshtein
from convertor import show_image

parser = argparse.ArgumentParser(description='Recipe digitalization')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--threads', type=int, default=1, help='Number of threads')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--show_prediction', default=False, action='store_true', help='Show predicted function')
parser.add_argument('--model', default=None, help="Load model")

subparsers = parser.add_subparsers(dest='command')
infer_img = subparsers.add_parser('infer', help='Inference images')
infer_img.add_argument('model_path', help='Create prediction with trained model')
infer_img.add_argument('--img_dir', default=None, help="Image directory")
infer_img.add_argument('--infer_train', default=False, action='store_true', help="Inference validation set")


def compute_metrics(eval_pred, tokenizer: T5Tokenizer):
    predictions, target_ans = eval_pred

    target_ans = np.where(target_ans != -100, target_ans, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(target_ans, skip_special_tokens=True)

    # bleu_labels = [[label] for label in decoded_labels]
    # bleu = bleu_score(decoded_preds, bleu_labels, n_gram=3).item()

    # Edit distance ratio
    ratio_sum = 0.0
    levanstein_9 = 0.0
    levanstein_8 = 0.0
    for pred, label in zip(decoded_preds, decoded_labels):
        r = Levenshtein.ratio(pred, label)
        ratio_sum += r
        levanstein_9 += (1.0 if r > 0.9 else 0.0)
        levanstein_8 += (1.0 if r > 0.8 else 0.0)

    return { #"bleu": bleu,
            "levenshtein_ratio": ratio_sum / len(decoded_preds),
            "levenshtein_above_0.9": levanstein_9 / len(decoded_preds),
            "levenshtein_above_0.8": levanstein_8 / len(decoded_preds)}


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
    model.generation_config.max_length = 4000

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
            PreprocessedRecipeDataset(tokenizer, json_file="PreprocessedDataset_2.json"),
            [0.9, 0.1],
            torch.Generator().manual_seed(1)
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer)
    )

    trainer.train()


def generate_prediction(args):
    finetuned_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    finetuned_model.generation_config.max_length = 4000

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    dataset = None
    if args.infer_train:
        dataset = RecipeDataset(generate_images=True)
    else:
        dataset = OnlyImageRecipeDataset(args.img_dir)

    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='en',
                    use_angle_cls=True
    )
    with open("prediction.txt", 'w') as f:
        for idx in range(len(dataset)):
            image, *targets = dataset[idx]
            for idx in range(len(targets)):
                if isinstance(targets[idx], list):
                    targets[idx] = "|".join(targets[idx])

            result = ocr.ocr(image, det=True, rec=True)[0]
            ocr_text, ocr_boxes = extract_data_from_ocr_result(result)
            ocr_text, ocr_boxes = merge_close_boxes(ocr_text, ocr_boxes, threshold=10)
            print(f"--- {idx} ---", file=f)
            for question_type in range(len(QUESTIONS)):
                inputs = prepare_question(question_type, ocr_text, ocr_boxes)
                print(inputs, file=f)

                inputs = tokenizer(inputs, return_tensors="pt")
                outputs = finetuned_model.generate(**inputs)
                print(compute_metrics((outputs, tokenizer(targets[question_type], return_tensors="pt")['input_ids']),
                                      tokenizer
                                      ), file=f)
                answer = tokenizer.decode(outputs[0])
                print(answer, file=f)
                print("--------", file=f, flush=True)
            show_image(image)


if __name__ == '__main__':
    args = parser.parse_args()
    args.command = "infer"
    args.model_path = "checkpoint-30500"
    args.infer_train = True
    if args.command == "infer":
        generate_prediction(args)
    else:
        main(args)

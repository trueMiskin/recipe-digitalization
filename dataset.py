from ast import literal_eval
import os
import pandas as pd
import numpy as np
import torch
import convertor
import argparse
from PIL import Image
import json

R_TITLE = 1
R_INGREDIENTS = 2
R_INSTRUCTIONS = 3
R_IMAGE_NAME = 4
R_CLEANED_INGREDIENTS = 5

parser = argparse.ArgumentParser(prog='Recipe dataset',
                                 description='Without parameters program show images of recipes.')
parser.add_argument('-p', '--use-paddle', default=True, action='store_true',
                    help='Use paddle OCR on images')


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file="Food Ingredients and Recipe Dataset with Image Name Mapping.csv", transform=None,
                 generate_images=False):
        self.data = pd.read_csv(csv_file)
        
        self.data['Ingredients'] = self.data['Ingredients'].apply(literal_eval)
        self.data['Cleaned_Ingredients'] = self.data['Cleaned_Ingredients'].apply(literal_eval)
        self.data = self.data.to_numpy()
        self.generate_images = generate_images

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        template = np.random.choice(convertor.TEMPLATES)
        title = self.data[idx][R_TITLE]
        ingredients = self.data[idx][R_CLEANED_INGREDIENTS]
        instructions = self.data[idx][R_INSTRUCTIONS]

        if type(instructions) is str:
            instructions = [instructions]

        text = template(title, ingredients, instructions)

        if not self.generate_images:
            return title, ingredients, instructions

        images = convertor.convert_to_image(text, format='markdown')
        img = np.hstack(images)
        
        # Output numpy array of a image: 0-255, HxWxC
        return img, title, ingredients, instructions


P_OCR_TEXT = "texts"
P_OCR_BOXES = "boxes"
P_TITLE = "title"
P_INGREDIENTS = "ingredients"
P_INSTRUCTIONS = "instructions"
QUESTIONS = ["What is a title: ", "List the ingredients: ", "Describe the instructions: "]

def prepare_question(question_type, ocr_text, ocr_boxes, include_box_data=False):
    question = QUESTIONS[question_type]
    for text, box in zip(ocr_text, ocr_boxes):
        left, upper, right, lower = box
        if include_box_data:
            question += f"[{left}{lower}] {text} "
        else:
            question += f"{text} "
    return question


class PreprocessedRecipeDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, json_file="PreprocessedDataset.json", include_box_data=False):
        self.data = json.load(open(json_file, 'r'))
        self.tokenizer = tokenizer
        self.include_box_data = include_box_data
    
    def __len__(self):
        return len(self.data) * 3 # 3 questions per recipe

    def __getitem__(self, idx):
        question_type = idx % 3
        idx = idx // 3  # each recipe has 3 entries
        ocr_text = self.data[idx][P_OCR_TEXT]
        ocr_boxes = self.data[idx][P_OCR_BOXES]
        ans = self.data[idx][ [P_TITLE, P_INGREDIENTS, P_INSTRUCTIONS][question_type] ]
        
        if question_type != P_TITLE:
            ans = '\n'.join(ans)

        output_text = prepare_question(question_type, ocr_text, ocr_boxes, self.include_box_data)

        model_inputs = self.tokenizer(output_text)
        labels = self.tokenizer(text_target=ans)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

class OnlyImageRecipeDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_names = []
        for file in sorted(os.listdir(image_folder)):
            self.image_names.append(file)

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        print(self.image_folder + "/" + self.image_names[idx])
        image = Image.open(self.image_folder + "/" + self.image_names[idx])
        # Output numpy array of a image: 0-255, HxWxC
        return np.asarray(image), "", [], []


def extract_data_from_ocr_result(result):
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]

    # use OCR regions
    bboxes = []
    for box in boxes:
        left, upper = box[0][0], box[0][1]
        right, lower = box[2][0], box[2][1]
        bboxes.append([left, upper, right, lower])

    return txts, bboxes


def merge_close_boxes(ocr_text, ocr_boxes, threshold=10):
    idx = 0
    while idx < len(ocr_boxes):
        txt = ocr_text[idx]
        left, upper, right, lower = ocr_boxes[idx]

        for i in range(idx):
            l, u, r, b = ocr_boxes[i]
            l -= threshold
            u -= threshold
            r += threshold
            b += threshold
            # Check if boxes overlap
            if not (right < l or left > r or lower < u or upper > b):
                # merge boxes
                ocr_boxes[i] = [min(left, l), min(upper, u), max(right, r), max(lower, b)]
                ocr_text[i] += ' ' + txt
                ocr_boxes.pop(idx)
                ocr_text.pop(idx)
                break
        else:
            idx+=1

    return ocr_text, ocr_boxes


def ocr_with_paddle(img):
    from paddleocr import PPStructure,draw_structure_result,save_structure_res
    from paddleocr import PaddleOCR, draw_ocr, PPStructure

    finaltext = ''
    font_path = 'simfang.ttf' # PaddleOCR

    if False:
        table_engine = PPStructure(show_log=True,
                                layout_score_threshold=0.3,
                                layout_nms_threshold=0.5,
                                table=False,
                                image_orientation=False,
                                ocr=True,
                                lang='en',
                                merge_no_span_structure=False,
            )

        save_folder = './output'
        result = table_engine(img)
        save_structure_res(result, save_folder, "output")

        for line in result:
            line.pop('img')
            print(line)

        from PIL import Image

        im_show = draw_structure_result(img, result,font_path=font_path)
        convertor.show_image(im_show)
        im_show = Image.fromarray(im_show)
        im_show.save(save_folder + '/result.jpg')
    else:
        ocr = PaddleOCR(
            lang='en',
            use_angle_cls=True)
        result = ocr.ocr(img, det=True, rec=True)

        # for i in range(len(result[0])):
        #     text = result[0][i][1][0]
        #     finaltext += ' '+ text

        from PIL import Image
        result = result[0]
        image = Image.fromarray(img).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        print(boxes)
        print(txts)
        # result_structure = [
        #     {"type": "title", "bbox": [*line[0], *line[2]], "res": "", "img_idx": 0, "score": 0.8120958805084229}
        #     for line in result]
        # im_show = draw_structure_result(img, result_structure,font_path=font_path)
        convertor.show_image(im_show)
        im_show = Image.fromarray(im_show)
        im_show.save('result.jpg')

        # return finaltext


def main(args):
    r = RecipeDataset(generate_images=True)
    # r = OnlyImageRecipeDataset("recipe_edited")

    only_images = True if not args.use_paddle else False
    
    # Run for concrete img
    # from PIL import Image
    # img = np.asarray(Image.open("recipe_edited/griddle-recipe-book-2_edited.jpg"))
    # ocr_with_paddle(img)
    # return

    for data in r:
        img, title, ingredients, instructions = data

        lu, ru, rd, ld = [233.0, 160.0], [1397.0, 165.0], [1397.0, 197.0], [233.0, 192.0]
        from PIL import Image
        i = Image.fromarray(img).crop((*lu, *rd))
        # convertor.show_image(np.asarray(i))
        if only_images:
            convertor.show_image(img)
            continue

        generated_text = ocr_with_paddle(img)
        print(generated_text)
        return


if __name__ == "__main__":
    main(parser.parse_args())
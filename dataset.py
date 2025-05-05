from ast import literal_eval
import pandas as pd
import numpy as np
import torch
import convertor
import torchvision
import argparse
from paddleocr import PPStructure,draw_structure_result,save_structure_res
from paddleocr import PaddleOCR, draw_ocr, PPStructure

TITLE = 1
INGREDIENTS = 2
INSTRUCTIONS = 3
IMAGE_NAME = 4
CLEANED_INGREDIENTS = 5

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
        title = self.data[idx][TITLE]
        ingredients = self.data[idx][CLEANED_INGREDIENTS]
        instructions = self.data[idx][INSTRUCTIONS]

        if type(instructions) is str:
            instructions = [instructions]

        text = template(title, ingredients, instructions)

        if not self.generate_images:
            return title, ingredients, instructions

        images = convertor.convert_to_image(text, format='markdown')
        
        return torchvision.transforms.functional.to_tensor(np.hstack(images)),\
            title, ingredients, instructions


def ocr_with_paddle(img):
    finaltext = ''
    font_path = 'simfang.ttf' # PaddleOCR

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
    return

    # ocr = PaddleOCR(
    #     lang='en',
    #     use_angle_cls=True)
    # result = ocr.ocr(img, det=True, rec=False)

    # for i in range(len(result[0])):
    #     text = result[0][i][1][0]
    #     finaltext += ' '+ text

    # from PIL import Image
    # result = result[0]
    # image = Image.fromarray(img).convert('RGB')
    # boxes = [line[0] for line in result]
    # # txts = [line[1][0] for line in result]
    # # scores = [line[1][1] for line in result]
    # # im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    # result_structure = [
    #     {"type": "title", "bbox": [*line[0], *line[2]], "res": "", "img_idx": 0, "score": 0.8120958805084229}
    #     for line in result]
    # im_show = draw_structure_result(img, result_structure,font_path=font_path)
    # convertor.show_image(im_show)
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

    # return finaltext


def main(args):
    r = RecipeDataset(generate_images=True)

    only_images = True if not args.use_paddle else False

    for data in r:
        img, title, ingredients, instructions = data
        img *= 255
        img = img.type(torch.uint8)
        img = img.permute(1, 2, 0).numpy()

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
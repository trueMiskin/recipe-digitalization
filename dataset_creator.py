
from paddleocr import PaddleOCR, draw_ocr
import convertor
from dataset import RecipeDataset
import numpy as np
import json

r = RecipeDataset(generate_images=False)
ocr = PaddleOCR(lang='en',
                    use_angle_cls=True
)

output = []
idx = 0
for title, ingredients, instructions in r:
    print(f"Processing recipe: {idx}")
    for template in convertor.TEMPLATES:
        input_text = template(title, ingredients, instructions, font='times')
        images = convertor.convert_to_image(input_text, format='markdown')
        image = np.hstack(images)
        
        result = ocr.ocr(image, det=True, rec=True)[0]
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]

        # use OCR regions
        bboxes = []
        for box in boxes:
            left, upper = box[0][0], box[0][1]
            right, lower = box[2][0], box[2][1]
            bboxes.append([left, upper, right, lower])

        output.append({
            "boxes": bboxes,
            "texts": txts,
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions,
        })

    idx += 1
    if idx % 5 == 0:
        print(f"Processed {idx} recipes.")
        json.dump(output, open(f"PreprocessedDataset-checkpoint.json", "w"))

json.dump(output, open("PreprocessedDataset.json", "w"))


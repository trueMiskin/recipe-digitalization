
from paddleocr import PaddleOCR, draw_ocr
import convertor
from dataset import RecipeDataset, extract_data_from_ocr_result
import numpy as np
import json
import os

r = RecipeDataset(generate_images=False)
ocr = PaddleOCR(lang='en',
                    use_angle_cls=True
)

if os.path.exists(f"PreprocessedDataset-checkpoint.json"):
    output = json.load(open(f"PreprocessedDataset-checkpoint.json", "r"))
else:
    output = []
idx = len(output) // 2

for title, ingredients, instructions in r:
    print(f"Processing recipe: {idx}")
    for template in convertor.TEMPLATES:
        input_text = template(title, ingredients, instructions, font='times')
        images = convertor.convert_to_image(input_text, format='markdown')
        image = np.hstack(images)
        
        result = ocr.ocr(image, det=True, rec=True)[0]
        txts, bboxes = extract_data_from_ocr_result(result)

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


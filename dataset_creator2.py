import json
from dataset import merge_close_boxes

dataset = json.load(open("PreprocessedDataset.json", 'r'))

new_dataset = []
for datapoint in dataset:
    ocr_text = datapoint["texts"]
    ocr_boxes = datapoint["boxes"]
    title = datapoint["title"]
    ingredients = datapoint["ingredients"]
    instructions = datapoint["instructions"]
    print("Before", len(ocr_boxes))

    ocr_text, ocr_boxes = merge_close_boxes(ocr_text, ocr_boxes, threshold=10)
    
    print("After", len(ocr_boxes))
    new_dataset.append({
        "texts": ocr_text,
        "boxes": ocr_boxes,
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions
    })
json.dump(new_dataset, open("PreprocessedDataset_2.json", 'w'))
        
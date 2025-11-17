import json

dataset = json.load(open("PreprocessedDataset.json", 'r'))

new_dataset = []
for datapoint in dataset:
    ocr_text = datapoint["texts"]
    ocr_boxes = datapoint["boxes"]
    title = datapoint["title"]
    ingredients = datapoint["ingredients"]
    instructions = datapoint["instructions"]
    print("Before", len(ocr_boxes))

    idx = 0
    while idx < len(ocr_boxes):
        txt = ocr_text[idx]
        left, upper, right, lower = ocr_boxes[idx]
        
        for i in range(idx):
            l, u, r, b = ocr_boxes[i]
            l -= 10
            u -= 10
            r += 10
            b += 10
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
    
    print("After", len(ocr_boxes))
    new_dataset.append({
        "texts": ocr_text,
        "boxes": ocr_boxes,
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions
    })
json.dump(new_dataset, open("PreprocessedDataset_2.json", 'w'))
        
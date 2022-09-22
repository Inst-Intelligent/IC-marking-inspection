# -*- coding: utf-8 -*-
import imageio
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from model import Model
from sample import Sample
from utils import imgproc
from utils.text_utils import encode_char, decode_and_depart


def show(image):
    
    if type(image) is Image.Image:
        display(image)
        
    elif type(image) is np.ndarray or type(image) is imageio.core.util.Array:
        if str(image.dtype).startswith('uint8'):
            display(Image.fromarray(image))
        elif str(image.dtype).startswith('float'):
            display(Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)))
        else:
            print(image)
            
    else:
        print(image)

if torch.cuda.is_available():
    torch.cuda.set_device(torch.device('cuda:1'))

canvas_size = 256

model = Model()
model.cuda()
model.eval()

data_dir = Path('data/test_img')
label_dir = Path('data/test_label')

for x in data_dir.iterdir():

    if not x.suffix == '.jpeg':
        continues

    labelText = (label_dir / x.stem).with_suffix('.json')
    if not labelText.exists():
        continue

    label = json.loads(labelText.read_text())
    sample = Sample(x.name, imageio.imread(str(x)), label)
    image = sample.img

    raw_h, raw_w = image.shape[:2]
    img_resized = cv2.resize(image, (canvas_size, canvas_size))
    size_heatmap = (canvas_size // 2, canvas_size // 2)
    sample.resize(size_heatmap)
    preview_map = cv2.resize(img_resized, size_heatmap)

    ratio_h = raw_h / canvas_size
    ratio_w = raw_w / canvas_size

    img = imgproc.normalizeMeanVariance(img_resized)
    img = torch.from_numpy(img).permute(2, 0, 1)   
    img = img.unsqueeze(0).cuda()  


    out = model(img, [sample])

    if out is None:
        continue
    else:
        angle_predict, angle_labels, char_predict, char_labels, num_char_boxes, samples_info = out


    angle_predict = angle_predict.argmax(dim=1).cpu().numpy()
    char_predict = char_predict.argmax(dim=1).cpu().numpy()
    
    pred_texts = decode_and_depart(char_predict, num_char_boxes)
    gt_texts = decode_and_depart(char_labels, num_char_boxes)

    print('=' * 80) 
    print(sample.name)
    show(preview_map)

    valid_text_boxes = samples_info[0]['text_boxes']
    for i, _ in enumerate(valid_text_boxes):
        if i >= len(pred_texts):
            continue
        x1, y1, x2, y2 = valid_text_boxes[i]
        cv2.rectangle(preview_map, (x1, y1), (x2, y2), (255,205,110), 2)
        cv2.rectangle(preview_map, (x1 - 5, y1 - 15), (x1 - 5 + (len(pred_texts[i]) * 8 ), y1 - 2), (129,184,223), -1)
        cv2.putText(preview_map, f'{pred_texts[i]}', (x1 - 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)

    show(preview_map)

    print(pred_texts)
    print(gt_texts)
#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np 
import torch
import torch.nn as nn
from pathlib import Path
import json
import imageio
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from model import AttentionActivationBranch
from collections import OrderedDict  

from utils import  imgproc
from utils.watershed import watershed
from utils.gaussian import GaussianTransformer
from dataset.sample import Sample

cuda = torch.cuda.is_available()

preview = False

# Load the pretrained paramters to the model.
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


class ItemLoader(object):

    def __init__(self, net, target_size = 256, get_filename = False):

        self.net = net
        self.gaussianTransformer = GaussianTransformer(imgSize = target_size // 2)
        self.target_size = target_size
        self.get_filename = get_filename
        
    def setItem(self,  data_loc, label_loc, angle = 0):
        
        self.name = data_loc.name
        
        # Read the image to memory.
        self.img = imageio.imread(data_loc)
        
        # The original size of the image.
        raw_h, raw_w = self.img.shape[:2]

        # Read and parse the Json annotation file.
        self.label_loc = label_loc
        self.label = json.loads(label_loc.read_text())

        # Create a simple instance
        sample = Sample(self.name, self.img, self.label)
        
        # Resize the sample, and update the coordinates of the boxes.
        self.sample = sample
        sample.rotate(angle)
        sample.resize((self.target_size // 2, self.target_size // 2))
        self.img = sample.img
        self.labeled_bboxes = self.parseLabel()


    def parseLabel(self):
        
        labeled_bboxes = []
        for box in self.sample.boxes:
            # Neglect the box of the entire chip area.
            if box.text == '$' :
                continue
            # For non-character markings, set the sequence length to zero.
            if box.text == '#' :
                length = 0
            else:
                length = len(box.text)
            # Array for a word: [sequence length，word box angle，word box coordinators].
            labeled_bboxes.append([length, box.text, box.angle, box.toList()])

        return labeled_bboxes

    def resize_img(self, image):
        return cv2.resize(image, (self.target_size, self.target_size))

    def resize_gt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))
    
    def resize_boxes(self, image):
        pass
    
    # Calculate the confidence score for the word box.
    def get_confidence(self, real_len, pursedo_len):
        if pursedo_len == 0:
            return 0.
        return (real_len - min(real_len, abs(real_len - pursedo_len))) / real_len
    
    # Generate pseudo-ground truths.
    def get_gt_label(self):

        # Initialize the confidence mask.
        self.confidence_mask = np.ones(self.img.shape[:2])
        
        character_bboxes = []
        words = []
        angles = []
        confidences = []

        for num_text, text, text_angle, text_bbox in self.labeled_bboxes:
            
            # Transfer the word-level annotations to character-level annotations.
            if num_text > 0:
                bboxes, region_scores, confidence = self.get_persudo_bboxes(num_text, text_angle, text_bbox)
                self.draw_confidence_mask(bboxes, confidence)
                character_bboxes.append(np.array(bboxes))
                confidences.append(confidence)
                angles.append(text_angle)
                word = '0' * num_text
                words.append(word)

            else:
                l,t,r,b=text_bbox
                self.draw_confidence_mask([np.array([[l,t],[r,t],[r,b],[l,b]],dtype=np.int32)], 0)


        region_scores = np.zeros(self.img.shape[:2], dtype=np.float32)
        affinity_scores = np.zeros(self.img.shape[:2], dtype=np.float32)
        
        # Mapping the 2d gaussian matrices to the character boxes.
        # The results will be the pseudo-attention map labels.
        if len(character_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,character_bboxes,words,angles)
        
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
            
        image = Image.fromarray(self.img)
        image_preview = image
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness = 32.0 / 255, saturation=0.5)(image)
        image = self.resize_img(np.array(image))
        image = imgproc.normalizeMeanVariance(np.array(image), mean = (0.485, 0.456, 0.406),
                                              variance = (0.229, 0.224, 0.225))
        
        
        # Resize the attention maps the the target size.
        region_scores = self.resize_gt(region_scores)
        affinity_scores = self.resize_gt(affinity_scores)
        confidence_mask = self.resize_gt(self.confidence_mask)
        
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        
        return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, \
                confidences, self.sample, (image_preview, region_scores, affinity_scores, confidence_mask) if preview == True else -1
    
    
    def get_persudo_bboxes(self, num_text, text_angle, text_bbox):
        
        # Set the boundary.
        l, t, r, b = text_bbox
        
        input = self.img[t:b, l:r].copy()
        input_h, input_w = input.shape[:2]

        # Rotate the word box to the upside direction.
        if text_angle > 0:
            input = cv2.rotate(input, 2 - text_angle // 90  + 1)

        # The size after the rotation.
        rotated_h, rotated_w = input.shape[:2]
        
        # Scale the box height to 64px, and scale the width in equally.
        scale = 64.0 / rotated_h
        input = cv2.resize(input, None, fx = scale, fy = scale)
        right_margin_res = 0 #从右侧加边距补齐
        
        # The size after the scaling.
        resized_h, resized_w = input.shape[:2]

        # Padding right boundary so that the width can be a multiple of 32.
        if resized_w % 32 != 0:
            right_margin_res = (resized_w // 32 + 1) * 32 - resized_w
        
        # Extract the backround color and do adaptive padding.
        bgc = np.array([np.argmax(cv2.calcHist([input], [i], None, [256], [0.0,255.0])) for i in range(3)])
        margin = 20
        feed = np.ones((resized_h + margin * 2, resized_w + margin * 2 + right_margin_res, 3), dtype=np.uint8) * bgc
        feed = feed.astype(np.uint8)
        feed[margin:margin + resized_h, margin:margin + resized_w,:] = input
        input = feed
        
        img_torch = torch.from_numpy(imgproc.normalizeMeanVariance(input, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)))
        img_torch = img_torch.permute(2, 0, 1).unsqueeze(0)
        img_torch = img_torch.float().cuda()
        
        # Get the attention maps of the word boxes.
        scores, _, _ = self.net(img_torch)
        region_scores = scores[0, :, :, 0].cpu().data.numpy() 
        region_scores = np.uint8(np.clip(region_scores, 0, 1) * 255) 
        
        region_scores = cv2.resize(region_scores, None, fx = 2, fy = 2) 
        region_scores_color = cv2.cvtColor(region_scores, cv2.COLOR_GRAY2BGR)
        
        # Character spotting for the word box.
        pseudo_boxes = watershed(region_scores_color, region_scores, low_text = 0.5)
        
        # Get the confidence for the word box.
        confidence = self.get_confidence(num_text, len(pseudo_boxes))
        bboxes = []
        
        # If the confidence<=0.5, divide the box equally into N regions.
        # N is the length of the annotaed word.
        if confidence <= 0.5:
            width = resized_w
            height = resized_h
            width_per_char = width / num_text
            for i in range(num_text):
                left = i * width_per_char
                right = (i + 1) * width_per_char
                bbox = np.array([[left, 0], [right, 0], [right, height],
                                 [left, height]])
                bboxes.append(bbox)
            bboxes = np.array(bboxes, np.float32)
            confidence = 0.5
        else:
             # remove the adaptive padding.
            bboxes = pseudo_boxes - np.array([margin, margin])
        bboxes = bboxes / scale
 
        # Mapping the predicted character boxes to the image space.
        bboxes = np.roll(bboxes, 4 - text_angle // 90, axis = 1)
        # If the angle is 90 or 270, swap the x and y coordinates.
        if text_angle % 180 == 90:
            bboxes = np.roll(bboxes, 1, axis = 2)

        expand_width = 1
        if text_angle % 180 == 0:
            # make a small expansion on the up-down side. 
            relative_corner = np.matrix([[l,t - expand_width],[l,t - expand_width],[l,t + expand_width],[l,t + expand_width]])
        else:
             # make a small expansion on the left-right side. 
            relative_corner = np.matrix([[l - expand_width,t],[l + expand_width,t],[l + expand_width,t],[l - expand_width,t]])
            
        # If the angle is 90 or 180, subtract the x coordinates by the box width.
        if 90 <= text_angle <= 180:
            bboxes[:, :, 0] = input_w - bboxes[:, :, 0]
        # If the angle is 180 or 270, subtract the y coordinates by the box height.
        if text_angle >= 180:
            bboxes[:, :, 1] = input_h - bboxes[:, :, 1]
            
        # The order of a character box: [top-left，top-right，bottom-right，bottom-left].
        for i in range(len(bboxes)):
            startidx = bboxes[i].sum(axis=1).argmin()
            bboxes[i] = np.roll(bboxes[i], 4 - startidx, 0)
            temp = np.matrix(bboxes[i])
            bboxes[i] = np.array(temp + relative_corner)

        bboxes[:, :, 0] = np.clip(bboxes[:, :, 0], 0., self.img.shape[1] - 1)
        bboxes[:, :, 1] = np.clip(bboxes[:, :, 1], 0., self.img.shape[0] - 1)
         
        # Arrange the character boxes by the distance to the origin point.
        bboxes = sorted(bboxes, key = lambda x:x[0][0] + x[0][1])
     
        return bboxes, region_scores, confidence

    def draw_confidence_mask(self, bboxes, confidence):
        
        for bbox in bboxes:
            cv2.fillPoly(self.confidence_mask, [np.int32(bbox)], (confidence))

# 数据集类，继承 Dataset 接口
class DetectionData(torch.utils.data.Dataset):
    
    def __init__(self, net, data_dir, label_dir, target_size=256, device=torch.device('cpu'), get_filename=False):
        
        torch.cuda.set_device(device)

        if isinstance(data_dir,str):
            data_dir = Path(data_dir)
        if isinstance(label_dir,str):
            label_dir = Path(label_dir)

        self.net = net
        self.net.eval()
        self.itemLoader = ItemLoader(net, target_size, get_filename)
        
        data_list = []
        for x in data_dir.iterdir():
            
            if not x.suffix == '.jpeg':
                continue
            labelText = (label_dir / x.stem).with_suffix('.json')
            if not labelText.exists():
                continue
            data_list += [[x, labelText]]
        
        self.data_list = data_list
        self.load_label()

    def load_label(self):
        self.data = []
        for i in range(len(self.data_list)):
            for angle in [0, 90, 180, 270]:
                self.itemLoader.setItem(*self.data_list[i], angle)
                self.data.append(self.itemLoader.get_gt_label())
    
    # dynamic label update mechanism
    def update_label(self):
        # ...
        for i in range(len(self.data)):
            if self.data[i][-1] < 1.0:
                self.itemLoader.setItem(*self.data_list[i])
                self.data[i] = self.itemLoader.get_gt_label()
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
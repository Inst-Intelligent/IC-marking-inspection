import numpy as np 
import cv2

import torch
from torchvision.ops import box_iou, roi_pool, roi_align
from torchvision.transforms.functional import rotate, resize

from functools import cmp_to_key
from scipy.ndimage.filters import gaussian_filter

from utils.gaussian import GaussianTransformer
from utils.watershed import watershed
from utils.text_utils import encode_char

size_heatmap = (128, 128)
text_box_iou_threshold = 0.30
link_threshold = 0.40
text_threshold = 0.55
low_text = 0.65
character_threshold = 0.3
similarity_threshold = 0.8

# Calculate the IoU for the boxes in list/nparray format
def iou(box1, box2):
    l1,t1,r1,b1 = box1
    l2,t2,r2,b2 = box2
    
    x1, x2 = max(l1,l2), min(r1,r2)
    y1, y2 = max(t1,t2), min(b1,b2)

    area1 = (b1 - t1) * (r1 - l1)
    area2 = (b2 - t2) * (r2 - l2)
    overlap = area((x1, y1, x2, y2))

    return overlap / (area1 + area2 - overlap)


# Word spotting and RoI pooling for a batch.
def feature_roi_pooling(attention_maps, shared_feature, global_feature, samples):

    feature = shared_feature.detach() 
    batch_num = attention_maps.size()[0]
    batch_text_angle_labels = [] 
    batch_pooled_features = []
    batch_sample_info = []

    for i in range(batch_num):
        score_text = attention_maps[i,:,:,0].cpu().data.numpy()
        score_link = attention_maps[i,:,:,1].cpu().data.numpy()
        text_boxes, text_angle_labels, sample_info = find_text_boxes(score_text, score_link, size_heatmap, samples[i])
        batch_sample_info.append(sample_info)
       
        # ROI Pooling
        box_features = torch.zeros([text_boxes.shape[0], 704, 16, 16 ]) 
        for j,box in enumerate(text_boxes):
            boxes = torch.Tensor([i, *box])
            boxes = torch.unsqueeze(boxes, 0)
            pooled_feature = roi_pool(feature, boxes.cuda(), 16) 
            image_feature = global_feature[i: i+1]
            box_features[j,:,:,:] = torch.cat([pooled_feature, image_feature], dim=1)
            batch_text_angle_labels += text_angle_labels
            batch_pooled_features.append(box_features)

    if len(batch_pooled_features) == 0:
        return None, None, None

    batch_pooled_features = torch.cat(batch_pooled_features, dim = 0)
    batch_text_angle_labels = torch.LongTensor(batch_text_angle_labels) // 90

    return batch_pooled_features.cuda(), batch_text_angle_labels, batch_sample_info

# Find the word boxes by attention maps.
def find_text_boxes(score_text, score_link, size_heatmap, sample):

    textmap = np.clip(score_text, 0, 1)
    linkmap = np.clip(score_link, 0, 1)
    _, textmap_s = cv2.threshold(textmap, text_threshold, 1, cv2.THRESH_BINARY)
    _, linkmap_s = cv2.threshold(linkmap, link_threshold, 1, cv2.THRESH_BINARY)
    
    # Find the bounding boxes of the connected areas.
    merged_map = np.clip(textmap_s + linkmap_s, 0, 1)
    merged_map = np.uint8(merged_map * 255)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(merged_map)

    predict_text_boxes = [] 
    padding = 0

    for x, y, w, h, _ in stats[1:]:
        predict_text_boxes.append([x, y, x + w, y + h])
    predict_text_boxes = np.array(predict_text_boxes)

    # If no word boxes have been found.
    if predict_text_boxes.shape[0] == 0:
        return np.array([]), [], None;

    # Transfer to a tensor, and calculate IoU with the ground truths.
    pboxes = torch.Tensor(predict_text_boxes)
    gboxes = torch.Tensor(sample.toResizedArray())
    text_ious = box_iou(pboxes, gboxes)
    text_match_score, text_match_index = torch.max(text_ious, dim = 1)
    text_match_filter = text_match_score > text_box_iou_threshold

    # Filter the valid boxes and get their indices.
    valid_predict_text_boxes = predict_text_boxes[text_match_filter.numpy()]
    valid_predict_text_boxes += np.array([-padding, -padding, padding, padding])
    valid_predict_text_boxes = np.clip(valid_predict_text_boxes, 0, size_heatmap[0])
    valid_text_match_index = text_match_index[text_match_filter]

    # No valid word boxes have been activated.
    if valid_predict_text_boxes.shape[0] == 0:
        return np.array([]), [], None;
    
    # Genearet direction labels and character labels.
    valid_text_gt_angle = [sample.angles()[i] for i in valid_text_match_index]
    valid_text_gt = list([encode_char(sample.texts()[i]) for i in valid_text_match_index])

    sample_info = dict()
    sample_info['text_boxes'] = valid_predict_text_boxes
    sample_info['text_labels'] = valid_text_gt
    sample_info['text_angles'] = valid_text_gt_angle
    return valid_predict_text_boxes, valid_text_gt_angle, sample_info


# Find the link points from a link map.
def find_link_points(heatmap):
    heatmap = gaussian_filter(heatmap, sigma=1)
    heatmap_left = np.zeros(heatmap.shape)
    heatmap_left[1:, :] = heatmap[:-1, :]
    heatmap_right = np.zeros(heatmap.shape)
    heatmap_right[:-1, :] = heatmap[1:, :]
    heatmap_up = np.zeros(heatmap.shape)
    heatmap_up[:, 1:] = heatmap[:, :-1]
    heatmap_down = np.zeros(heatmap.shape)
    heatmap_down[:, :-1] = heatmap[:, 1:]

    peaks_binary = np.logical_and.reduce((heatmap >= heatmap_left, heatmap >= heatmap_right, heatmap >= heatmap_up, heatmap >= heatmap_down, heatmap>0.25))
    return peaks_binary.astype(np.uint8)

# Calculate the area of a box.
def area(box):
    x1, y1, x2, y2 = box
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return w * h

# Group the character boxes, by finding their matched word boxes.
def group_charbox(charboxes, textboxes, character_threshold = 0.3):

    scores = np.zeros((charboxes.shape[0], textboxes.shape[0]))
    for i, (cx1, cy1, cx2, cy2) in enumerate(charboxes):
        for j, (tx1, ty1, tx2, ty2) in enumerate(textboxes):
            x1, y1 = max(cx1, tx1), max(cy1, ty1)
            x2, y2 = min(cx2, tx2), min(cy2, ty2)
    
            scores[i, j] = area((x1, y1, x2, y2)) / area((cx1, cy1, cx2, cy2))

    max_scores = np.max(scores, axis = 1)
    max_index = np.argmax(scores, axis = 1)

    return charboxes[max_scores >= character_threshold], max_index[max_scores >= character_threshold]


# Arrange the character boxes
def arrange_chars_in_box(box1, box2):

    # If the boxes are from different words, sort by their group indices.
    if box1[2] != box2[2]:
        return box1[2] - box2[2]

    # Calculate the comparison indicator.
    xmin1,ymin1,xmax1,ymax1 = box1[0]
    xmin2,ymin2,xmax2,ymax2 = box2[0]
    
    m = (ymin1 - ymax2) * (ymax1 - ymin2)
    if m<0 :
        comp = xmin1 - xmin2
    else:
        comp = ymin1 - ymin2
    
    if box1[1] == box2[1] and box1[1] >= 2:
        comp = - comp

    return comp

# NMS for eliminating the overlapping boxes.
def nms_charbox(charboxes, angles, groups, highest_score = 0.5):

    markers = np.ones(charboxes.shape[0])

    for i, box1 in enumerate(charboxes):
        for j, box2 in enumerate(charboxes):
            if i == j or markers[i] == 0 or markers[j] == 0:
                continue

            ovr = iou(box1, box2)
            if ovr > highest_score:
                removed_idx = i if area(box1) < area(box2) else j
                markers[removed_idx] = 0

    return charboxes[markers == 1], angles[markers == 1], groups[markers == 1]


# Filter the candidate boxes.
def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) 
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
gaussian_kernel2d = GaussianTransformer()._gen_gaussian_heatmap(16, 1.6).astype(np.float32) / 255.0
penalty_factor = lambda x: 2 * np.sin(0.5 * np.pi * x)
std_vec = np.array([1.0] * 8 + [-1.0] * 8)
def box_score(text_map_box):
    sum_hor = np.sum(text_map_box, axis = 0)
    sum_ver = np.sum(text_map_box, axis = 1)
    del_hor = np.sign(sum_hor[1:] - sum_hor[:-1]) - std_vec
    del_ver = np.sign(sum_ver[1:] - sum_ver[:-1]) - std_vec
    del_hor[del_hor == 0]
    return np.sum((del_hor / 2 + penalty_factor(del_hor)) ** 2 + 
        (del_ver / 2 + penalty_factor(del_ver)) ** 2) / 8


# Character box refinement.
def fix_char_box(charboxes, groups, angles, textmap, linkmap, size_heatmap, margin_ls = 4, margin_ss = 0):

    global gaussian_kernel2d
    
    segmap = find_link_points(linkmap)

    out_boxes = list()
    out_angles = list()
    out_groups = list()

    for i, (x1, y1, x2, y2) in enumerate(charboxes):
        x1, x2 = max(0, x1), min(size_heatmap[0], x2)
        y1, y2 = max(0, y1), min(size_heatmap[1], y2)

        # Check if the box is horizontal.
        horizontal = True if angles[i] % 2 == 0 else False
        segbox = segmap[y1: y2, x1: x2]
        

        # Get x coordinates for horizontal boxes.
        # Get y coordinates for vertical boxes.
        points = np.where(segbox == 1)[1] if horizontal else np.where(segbox == 1)[0]
        points += x1 if horizontal else y1

        # If no link points in the box, skip the filtering.
        if points.shape[0] == 0:
            out_boxes.append([x1, y1, x2, y2 ])
            out_angles.append(angles[i])
            out_groups.append(groups[i])
            continue

        points.sort()
        points = np.insert(points, 0, x1 if horizontal else y1)
        points = np.append(points, x2 if horizontal else y2)

        span = points[1:] - points[:-1]

        for j in range(len(span)):

            if span[j] <= 0:
                continue

            st = points[j]
            ed = points[j + 1]
            
            if horizontal:
                xx1, yy1, xx2, yy2 = st - margin_ss, y1 - margin_ls, ed + margin_ss, y2 + margin_ls
            else:
                xx1, yy1, xx2, yy2 = x1 - margin_ls, st - margin_ss, x2 + margin_ls, ed + margin_ss

            text_map_box = cv2.resize(textmap[yy1: yy2, xx1: xx2], (17,17))
            score = box_score(text_map_box)

            if score < similarity_threshold:
                out_boxes.append([xx1, yy1, xx2, yy2])
                out_angles.append(angles[i])
                out_groups.append(groups[i])
    
    if len(out_groups) == 0:
        return np.array([]), np.array([]), np.array([])

    zipped_boxes = list(zip(out_boxes, out_angles, out_groups))
    zipped_boxes.sort(key = cmp_to_key(arrange_chars_in_box))
    out_boxes, out_angles, out_groups = zip(*zipped_boxes)
    out_boxes = np.array(out_boxes).clip(0, size_heatmap * 2)
    out_angles, out_groups  = np.array(out_angles), np.array(out_groups)
    out_boxes, out_angles, out_groups = nms_charbox(out_boxes, out_angles, out_groups)

    return out_boxes, out_angles, out_groups

# Spot the character boxes and do RoI align for a sample.
def feature_roi_align(feature, valid_text_boxes, valid_text_labels, angles, score_text, score_link):

    textmap = cv2.resize(np.clip(score_text, 0, 1), size_heatmap)
    linkmap = cv2.resize(np.clip(score_link, 0, 1), size_heatmap)
    textmap_color = cv2.cvtColor(textmap * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    # Spot the character boxes by watershed.
    predict_char_boxes_ = watershed(textmap_color, textmap_color, low_text)
    predict_char_boxes = [[*box[0], *box[2]] for box in predict_char_boxes_]
    predict_char_boxes = np.array(predict_char_boxes).astype(np.int)

    if len(predict_char_boxes) == 0:
        return None

    # Group the character boxes, and match the boxes to the word boxes.
    valid_char_boxes, valid_char_group = group_charbox(predict_char_boxes, valid_text_boxes)
    valid_char_angles = np.array([angles[group] for group in valid_char_group])
    valid_char_boxes, valid_char_angles, valid_char_group = fix_char_box(valid_char_boxes, valid_char_group, valid_char_angles, textmap, linkmap, size_heatmap)

    if valid_text_labels is not None:
        text_label_filter = np.zeros(valid_text_boxes.shape[0]).astype(np.int)
        if len(valid_char_group) > 0:
            text_label_filter[valid_char_group] = 1
        
        valid_num_char_boxes, valid_char_labels, valid_num_char_labels = [], [], []

        valid_char_filter = np.ones(valid_char_boxes.shape[0]).astype(np.int)
        for i, x in enumerate(text_label_filter):
            if x == 1:
                valid_num_char_boxes += [len(valid_char_group[valid_char_group == i])]
                valid_char_labels += valid_text_labels[i]
                valid_num_char_labels += [len(valid_text_labels[i])]
                
        valid_char_boxes = valid_char_boxes[valid_char_filter == 1]

    else:
        valid_char_labels, valid_num_char_labels = None, None

    box_features = torch.zeros([valid_char_boxes.shape[0], 195, 16, 16 ]) # N*C*W*H
    for i, box in enumerate(valid_char_boxes):
        boxes = torch.Tensor([0, *box])
        boxes += torch.Tensor([0, -1, -1, 1, 1]) 
        boxes = torch.unsqueeze(boxes, 0)
        aligned_feature = roi_align(feature, boxes.cuda(), 16)[0]
        aligned_feature = rotate(aligned_feature, int(valid_char_angles[i] * 90))
        box_features[i,:,:,:] = aligned_feature

    return box_features, valid_char_boxes, valid_num_char_boxes, valid_char_angles, valid_char_labels, valid_num_char_labels

# Spot the character boxes and do RoI align for a batch.
def batch_feature_roi_align(features, batch_sample_info, attention_maps, images):


    batch_num_char_boxes = []
    batch_box_featuers = []
    batch_char_labels = []
    batch_num_char_labels = []

    images = resize(images, (128, 128))
    features = torch.cat([features, images], dim = 1)

    for i, sample_info in enumerate(batch_sample_info):

        if sample_info is None:
            continue

        score_text = attention_maps[i,:,:,0].cpu().data.numpy()
        score_link = attention_maps[i,:,:,1].cpu().data.numpy()
        
        roi_result = feature_roi_align(features[i: i+1], 
            sample_info['text_boxes'], 
            sample_info['text_labels'], 
            np.array(sample_info['text_angles'])  // 90, 
            score_text,
            score_link)
        
    
        if roi_result is None:
            continue
        
        box_features, valid_char_boxes, valid_num_char_boxes, valid_char_angles, valid_char_labels, valid_num_char_labels = roi_result
        batch_num_char_boxes += valid_num_char_boxes
        batch_char_labels += valid_char_labels
        batch_num_char_labels += valid_num_char_labels

        batch_box_featuers.append(box_features)

    if len(batch_box_featuers) == 0:
        return None, None, None, None

    batch_char_labels, batch_num_char_labels = torch.tensor(batch_char_labels), torch.LongTensor(batch_num_char_labels)
    batch_aligned_features = torch.cat(batch_box_featuers, dim = 0)
   
    return batch_aligned_features.cuda(), batch_num_char_boxes, batch_char_labels, batch_num_char_labels 
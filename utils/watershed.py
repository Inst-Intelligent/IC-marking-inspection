import cv2
import numpy as np
import math
import Polygon as plg
from PIL import Image

def watershed(oriimage, image, low_text=0.6):
    # viz = True
    boxes = []
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    ret, binary = cv2.threshold(gray, 0.3 * np.max(gray), 255, cv2.THRESH_BINARY)

    # Eliminate the noises.
    kernel = np.ones((3, 3), np.uint8)
    mb = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=3)  # iterations连续两次开操作
    sure_bg = cv2.dilate(mb, kernel, iterations=3)  # 3次膨胀,可以获取到大部分都是背景的区域
    sure_bg = np.uint8(mb)

    ret, sure_fg = cv2.threshold(gray, low_text * gray.max(), 255, cv2.THRESH_BINARY)
    surface_fg = np.uint8(sure_fg) 
    unknown = cv2.subtract(sure_bg, surface_fg)
    
    # Find the seed areas.
    ret, markers = cv2.connectedComponents(surface_fg)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(surface_fg, connectivity=4)
    markers = labels.copy() + 1
    markers[unknown == 255] = 0
    
    # Employ the CV watershed.
    markers = cv2.watershed(oriimage, markers=markers)
    oriimage[markers == -1] = [0, 0, 255]
    
    color_markers = np.uint8(markers + 1)
    color_markers = color_markers / (color_markers.max() / 255)
    color_markers = np.uint8(color_markers)
    color_markers = cv2.applyColorMap(color_markers, cv2.COLORMAP_JET)

    for i in range(2, np.max(markers) + 1):
        markers2 = np.zeros(markers.shape,dtype=np.uint8)
        markers2[markers==i]=255
        markers2 = cv2.dilate(markers2, kernel, iterations=3)
        np_contours = np.roll(np.array(np.where(markers2 == 255)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        poly = plg.Polygon(box)
        area = poly.area()
        if area < 10:
            continue
        box = np.array(box)
        boxes.append(box)
    return np.array(boxes)
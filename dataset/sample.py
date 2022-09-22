import cv2
import imageio
import json

from pathlib import Path
from PIL import Image
import numpy as np

class BBox(object):
    
    def __init__(self, box):
        
        self.x1 = box['coordinates']['x'] - box['coordinates']['width'] / 2
        self.y1 = box['coordinates']['y'] - box['coordinates']['height'] / 2
        self.x2 = box['coordinates']['x'] + box['coordinates']['width'] / 2
        self.y2 = box['coordinates']['y'] + box['coordinates']['height'] / 2
        
        self.x1_, self.y1_, self.x2_, self.y2_ = self.x1, self.y1, self.x2, self.y2
        
        # For non-character markings, the text and angle will be not recorded, 
        # and also, it will be regarded as a invalid box.
        if box['label'] == '#' or box['label'] == '$':
            self.text = box['label']
            self.angle = None
            self.valid = False
            
        # The ground-truth sequence and the angle are jointed with '/' in the annotation
        else:
            text,angle = box['label'].split('/')
            self.text = text
            self.angle = int(angle)
            self.valid = True
    
    # Invoked by Sample.rotate, this method will be invoked.
    # M is the perspective transform matrix
    def rotate(self, M, angle):
        self.x1, self.y1 = M.dot(np.array([self.x1, self.y1, 1]))
        self.x2, self.y2 = M.dot(np.array([self.x2, self.y2, 1]))
        
        if angle == 180 or angle == 90:
            self.x1, self.x2 = self.x2, self.x1
        if angle == 270 or angle == 180:
            self.y1, self.y2 = self.y2, self.y1
            
        if self.angle is not None:
            self.angle = (self.angle + angle) % 360
        
    def toList(self):
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]
    
    def toResizedList(self):
        return [int(self.x1_), int(self.y1_), int(self.x2_), int(self.y2_)]
        
class Sample(object):
    
    def __init__(self, name, img, label):
        self.name = name
        self.img = img.copy()
        self.height, self.width = img.shape[:2]
        self.boxes = []
        for box in label[0]['annotations']:
            self.boxes.append(BBox(box))
    
    def show(self):
        display(Image.fromarray(self.img))
        
    def drawLabel(self):
        for box in self.boxes:
            color = (0,0,255) if box.text == '$' else (0,255,255)
            box.x1, box.y1, box.x2, box.y2 = np.array([box.x1, box.y1, box.x2, box.y2]).astype(np.int)
            cv2.rectangle(self.img, (box.x1, box.y1), (box.x2, box.y2), color, 1)
     
    # If the sample is augmented to another direction, this method should be executed.
    def rotate(self, angle = 0):
        if angle == 0:
            return
        M = cv2.getRotationMatrix2D((self.width//2, self.height//2), -angle, 1)
        if angle == 90 or angle == 270:
            self.width, self.height = self.height, self.width
            # 校正中心点
            M[0, 2] += (self.width - self.height) // 2
            M[1, 2] += (self.height - self.width) // 2
            
        self.img = cv2.warpAffine(self.img, M, (self.width, self.height))
        for box in self.boxes:
            box.rotate(M, angle)
    
    # If the sample image has been resized, this method should be manually invoked.
    def resize(self, tar_size):
        for box in self.boxes:
            ws, hs = (self.width, self.height)
            wt, ht = tar_size
            box.x1_ = wt / ws * box.x1
            box.y1_ = ht / hs * box.y1
            box.x2_ = wt / ws * box.x2
            box.y2_ = ht / hs * box.y2
    
    
    def toArray(self):
        return np.array([box.toList() for box in filter(lambda box: box.valid is True,self.boxes) ])

    def toResizedArray(self):
        return np.array([box.toResizedList() for box in filter(lambda box: box.valid is True,self.boxes) ])
    
    def angles(self):
        return np.array([box.angle for box in filter(lambda box: box.valid is True,self.boxes) ])

    def texts(self):
        return np.array([box.text for box in filter(lambda box: box.valid is True,self.boxes)])

if __name__ == '__main__':
    
    data_dir = Path('valData')
    label_dir = Path('valLabel')

    for x in data_dir.iterdir():

        if not x.suffix == '.jpeg':
            continue

        img = imageio.imread(str(x))

        labelText = (label_dir / x.stem).with_suffix('.json')

        if not labelText.exists():
            continue

        label = json.loads(labelText.read_text())

        print(x.name)
        sample = Sample(x.name, img, label)
        sample.drawLabel()
        sample.show()

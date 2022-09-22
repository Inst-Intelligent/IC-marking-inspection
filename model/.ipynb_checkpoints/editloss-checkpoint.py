import torch
import torch.nn as nn
from utils import distance_and_trace

class EditLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.cross_entropy = nn.CrossEntropyLoss(reduction = 'none')
        
    def forward(self, x, y, num_chars, num_labels):
        
        pnt_x, pnt_y = 0, 0
        total = 0
        _, pred = torch.max(x, 1)
        attenuation = 0

        for i in range(len(num_chars)):
            
            _, trace = distance_and_trace(pred[pnt_x: pnt_x + num_chars[i]], y[pnt_y: pnt_y + num_labels[i]])
            trace = torch.tensor(list(trace)) 

            if trace.size(0) == 0:
                attenuation += 1
                continue
            
            x_, y_, alpha = x[trace[:,0].long() - 1 + pnt_x], y[trace[:,1].long() - 1 + pnt_y], trace[:, 2]
            pnt_x, pnt_y = pnt_x + num_chars[i], pnt_y + num_labels[i]
            total += torch.mean(self.cross_entropy(x_, y_)) 

        return total / (len(num_chars) - attenuation)
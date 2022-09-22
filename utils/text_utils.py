
_DICTIONARY = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"  

def encode_char(str):
    return [_DICTIONARY.find(x.upper()) for x in str]

def decode_char(ls):
    return ''.join([_DICTIONARY[x] for x in ls])

def decode_and_depart(vec, num_boxes):
    text = decode_char(vec)
    res = []
    pnt = 0
    for num in num_boxes:
        res.append(text[pnt: pnt + num])
        pnt += num
    return res


def metric_scores(pred_texts, labels):

    distance_sum = 0.
    precision_sum = 0.
    recall_sum = 0.
    
    for i in range(len(pred_texts)):
        distance,trace = distance_and_trace(pred_texts[i], labels[i])
        trace = np.array(list(trace))
        trace = trace[:,2] if trace.shape[0] > 0 else trace
        acc = np.sum(trace[trace == 1.])
        distance_sum += distance
        precision_sum += acc / len(pred_texts[i])
        recall_sum += acc / len(labels[i])
    
    return distance_sum, precision_sum / len(pred_texts), recall_sum / len(pred_texts)

def distance_and_trace(pred, label):
        
        trace = set()

        if torch.is_tensor(pred):
            npr, nl = pred.size(0), label.size(0)
        else:
            npr, nl = len(pred), len(label)
        
        # The case the sequences are of equal size.
        # Only for training.
        if  torch.is_tensor(pred) and npr == nl:
            for i in range(1, npr + 1):
                trace.add((i, i, 1))
            return 0, trace

        # ED matrix.
        stm = torch.zeros((npr + 1, nl + 1))
        som = torch.zeros((npr + 1, nl + 1))

        for i in range(1, npr + 1):
            stm[i, 0] = i
            som[i, 0] = 1

        for j in range(1, nl + 1):
            stm[0, j] = j
            som[0, j] = 1

        for i in range(1, npr + 1):
            for j in range(1, nl + 1):
                is_equal = char_equal(pred[i -1], label[j - 1])
                ops = [stm[i-1, j] + 1, stm[i, j - 1] + 1, stm[i-1, j-1] + (0 if is_equal else 1)] # 三种操作
                stm[i, j] = min(ops)
                
                if stm[i, j] == ops[2]:
                    som[i, j] = 1
                else:
                    som[i,j] = -1

        # Find the entrance
        x, y = torch.argmin(stm[:,-1]).item(), nl

        # Find the shortest path.
        while x >= 1 and y >= 1:
            if som[x, y] > 0:
                trace.add((x, y, som[x, y].item()))
            nex = min(stm[x - 1, y - 1], stm[x - 1, y], stm[x, y -1])
            if nex == stm[x - 1, y - 1]:
                x, y = x - 1, y - 1
            elif nex == stm[x - 1, y]:
                x, y = x - 1, y
            else:
                x, y = x, y - 1

        return stm[npr, nl], trace

def char_equal(a, b):

    if a == b:
        return True
    
    if torch.is_tensor(a):
        a,b = decode_char([a, b])

    # These characters can also be considered as the same.
    pairs = ['1I','5S', '0O']
    if f'{a}{b}' in pairs or f'{b}{a}' in pairs:
        return True
    
    return False


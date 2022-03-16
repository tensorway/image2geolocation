import cv2
import math 
import numpy as np
import torch as th

def great_circle_distance_relative(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    φ1 = lat1 * math.pi/180; # φ, λ in radians
    φ2 = lat2 * math.pi/180
    Δφ = (lat2-lat1) * math.pi/180
    Δλ = (lon2-lon1) * math.pi/180

    a = math.sin(Δφ/2) * math.sin(Δφ/2) + \
            math.cos(φ1) * math.cos(φ2) * \
            math.sin(Δλ/2) * math.sin(Δλ/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return c

def great_circle_distance(point1, point2):
    R = 6371; # km
    c = great_circle_distance_relative(point1, point2)
    d = R * c; # in km
    return d

def load_model(model, path):
    try:
        model.load_state_dict(th.load(path))
        print(f"loaded model ({type(model).__name__}) from {path}")
    except:
        print(f"could not load model ({type(model).__name__}) from {path}")

def save_model(model, path):
    th.save(model.state_dict(), path)

def seed_everything(seed: int):
    '''
    sets the random seed of everything to 
    make everything reproducable
    '''
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def draw_prediction(latitude, longitude, croatia_map=None, color=(0, 0, 255), img_path='assets/croatia_map.png'):
    '''
    takes a point (latitude and longitude) and
    draws it on a numpy image
    '''
    if croatia_map is None:
        croatia_map = cv2.imread(img_path)
    x1 = 0.0
    x2 = float(np.shape(croatia_map)[0])
    y1 = float(np.shape(croatia_map)[1])
    y2 = 0.0

    a1 = 13.184
    a2 = 19.512
    b1 = 42.139
    b2 = 46.582

    #functions to convert coordinates from geographical coordinates to dimensions of pictures
    def _f(x):
        return round((a2-a1)*(x-x1)/(x2-x1)+a1, 2)
    def _g(y):
        return round((b2-b1)*(y-y1)/(y2-y1)+b1, 2)
    def f_inv(x):
        return int((x2-x1)*(x-a1)/(a2-a1)+x1)
    def g_inv(y):
        return int((y2-y1)*(y-b1)/(b2-b1)+y1)

    croatia_map = cv2.circle(croatia_map, (f_inv(float(longitude)), g_inv(float(latitude))), 
                       radius=7, color=color, thickness=-1)
    return croatia_map


def contrastive_labels_mat(batch_size):
    '''
    generates a matrix in blocks of ones
    and with 0s on the diagonal
    example for a batch size of 3
    b 0 0
    0 b 0
    0 0 b 

    where b (block):
    0 1 1 1
    1 0 1 1
    1 1 0 1
    1 1 1 0 
    divided by 3 (to keep the sum of probabilities=1)
    '''
    mat = []
    for i in range(batch_size):
        for di in range(4):
            row = []
            for j in range(batch_size*4):
                if 4*i <= j < 4*i+4 and 4*i+di!=j:
                    row.append(1/3)
                else:
                    row.append(0)
            mat.append(row)
    return th.tensor(mat, dtype=th.float32)
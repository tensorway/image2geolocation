import math 
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
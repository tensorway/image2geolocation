import cv2
import math 
import numpy as np
import torch as th
import matplotlib.pyplot as plt


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
    except Exception as e:
        print(e)
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

def draw_prediction(latitude, longitude, croatia_map=None, color=(0, 0, 255), img_path='assets/croatia_map.png'):
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


def draw_gaussian_prediction(mu, cov, croatia_map=None, color=(0, 0, 255), img_path='assets/croatia_map.png'):
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


    w, v = np.linalg.eig(cov)
    sorted_idxs = np.argsort(w)
    v = v[:, sorted_idxs]
    w = w[sorted_idxs]

    alpha = math.acos(v[1, 0])
    alpha = math.degrees(alpha)

    latitude, longitude = mu

    center = np.array([f_inv(float(longitude)), g_inv(float(latitude))])

    def get_axis_len(i, significance):
        mayor_line = mu + w[i]*v[:, i]*significance
        mayor_line = np.array([ f_inv(mayor_line[1]), g_inv(mayor_line[0]) ]) - center
        return math.sqrt(np.sum(mayor_line**2))


    probs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    xis   = [0.02, 0.103, 0.211, 0.575, 1.386, 2.77, 4.61, 5.99, 9.21]

    for percentage, xi in zip(probs, xis):
        croatia_map = cv2.ellipse(
            croatia_map, 
            center, 
            (int(get_axis_len(0, xi)), int(get_axis_len(1, xi))), 
            alpha, 
            0, 
            360, 
            (255, 0, 0),
            2
            )
    plt.imshow(croatia_map)
    return croatia_map, probs


def mul2gaussians(m1, m2, c1, c2):
    c3 = c1 @ th.inverse(c1+c2) @ c2
    m3 = c2 @ th.inverse(c1+c2) @ m1 + c1 @ th.inverse(c1+c2) @ m2
    return m3, c3

def multiply_and_normalize_gaussians(mus, covs):
    m1, c1 = mus[0], covs[0]
    for m2, c2 in zip(mus[1:], covs[1:]):
        m1, c1 = mul2gaussians(m1, m2, c1, c2)
    return m1, c1
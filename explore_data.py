import cv2
import numpy as np
from tqdm import tqdm
from dataset import Image2GeoDataset

CROATIA_IMAGE_PATH = 'croatia_map.png'
RESULT_IMAGE_PATH = 'results/data_map.png'
SOUTH_BOUNDARY = 42 + 24/60 #23/60 is sea
NORTH_BOUNDARY = 46 + 33/60
EAST_BOUNDARY  = 19 + 27/60
WEST_BOUNDARY  = 13 + 30/60
# boundaries of croatia 42° 24′ i 46° 33′ sjeverne geografske širine, te 13° 30′ i 19° 27′


# go through every example and mark it on a geographic map
# save the resulting image TO RESULT_IMAGE_PATH
img = cv2.imread(CROATIA_IMAGE_PATH)
dataset = Image2GeoDataset()
for idx in tqdm(range(len(dataset))):
    sample = dataset[idx]
    lat, lon = sample['labels']
    i = int( (1-(lat-SOUTH_BOUNDARY)/(NORTH_BOUNDARY-SOUTH_BOUNDARY)) * img.shape[0] )
    j = int( (lon-WEST_BOUNDARY)/(EAST_BOUNDARY-WEST_BOUNDARY) * img.shape[1] )
    img[i, j] = np.array([255, 0, 0])

cv2.imwrite(RESULT_IMAGE_PATH, img)

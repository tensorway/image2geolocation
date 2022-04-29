import filecmp
import glob
from tqdm import tqdm

DATASET_PATH = 'dataset/lumen-datasci-2022-train/'
TRASH_IMG_PATH = 'dataset/trash_img.png'
# TRASH_IMG_PATH = 'dataset/example.jpg'

cnt = 0
num = 0
for file_name in tqdm(glob.glob(DATASET_PATH+'**/**/*')):
    if filecmp.cmp(TRASH_IMG_PATH, file_name, shallow=False):
        cnt += 1
    num += 1

print("trash=", cnt, "      out of", num, "images or", cnt/num*100, "%")

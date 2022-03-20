# %%

#sudo apt install tesseract-oct
#sudo apt install libtesseract-dev
#pip install pytesseract
#pip install tesseract-ocr

import pandas as pd
from PIL import Image
from tqdm import tqdm
import pytesseract

df= pd.read_csv('/home/mateo/lumen/image2geolocation/dataset/data.csv')
df.head()
# %%
#add column name
df["name"] = ""

#borders of cropped picture
left = 550
top = 625
right = 640
bottom = 640

#other authors part
count = 0
count_rate = 0.0

#total number of folders in data
NUM = 16000

loop = tqdm(range(NUM))

for i in loop:
    google = False
    for j in range(4):
        PATH = '/home/mateo/lumen/image2geolocation/dataset/data/' + df.uuid[i] + '/' + str(j*90) + '.jpg'
        im = Image.open(PATH)
        
        #crop image
        im_crop = im.crop((left, top, right, bottom))
        
        #resize image
        basewidth = 250
        wpercent = (basewidth/float(im_crop.size[0]))
        hsize = int((float(im_crop.size[1])*float(wpercent)))
        im_crop_and_resize = im_crop.resize((basewidth,hsize), Image.ANTIALIAS)
        
        #convert image to text with pytesseract
        text = pytesseract.image_to_string(im_crop_and_resize)
        
        #iterate through all 4 pictures in folder and remember first 
        #or the last one with non-empty text after conversion
        if(j == 0 or text != ''):
            text_it = text
        
        #if substring 'Goo' is found, then we can be sure ocr has found 'Google'
        if(text.find('Goo') != -1):
            google = True
    
    #if ocr is sure picture is Google, write it in dataframe
    if(google):
        count += 1
        count_rate = 1-count/(i+1)
        df.loc[i,'name'] = 'google'
    #else write last non-empty string, if one exists
    else:
        df.loc[i,'name'] = text_it
    loop.set_description(f"{count_rate: 4.3f}")

df.to_csv('data_with_names.csv')
#%%

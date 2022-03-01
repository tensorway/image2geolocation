import csv
import cv2
import numpy as np
from random import randrange
from matplotlib import pyplot as plt
from utils import great_circle_distance



#open csv file and choose random item row from it
file = open("dataset/data.csv")
csvreader = csv.reader(file)
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)
    

def pick_image_and_label():
    pick = rows.__getitem__(randrange(len(rows)))

    #load random picture from dataset
    fig = plt.figure(figsize=(10, 10))
    for i in range(4):
        filename = 'dataset/data/' + pick[0] + '/' + str(i*90) + '.jpg'
        im=cv2.imread(filename)
        fig.add_subplot(2, 2, i+1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(str(i*90) + " degrees")
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    imgs_guess = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    return pick, imgs_guess


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, param):
    croatia_map, pick, state = param
    
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

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates on terminal
        print('guess coordinates=', _f(x), ' ',_g(y))

        # displaying the circle of OUR GUESS
        croatia_map = cv2.circle(croatia_map, (x,y), 
                           radius=7, color=(255, 0, 0), thickness=-1)
        cv2.imshow('guess', croatia_map)

        # displaying the coordinates on terminal
        print('true coordinates', round(float(pick[2]), 2), round(float(pick[1]),2))
        distance = great_circle_distance([_g(y), _f(x)], (float(pick[1]), float(pick[2])))
        
        state['cumulative_distance'] += distance
        state['n_tries'] += 1
        print('distance=', distance, 'average_dist=', state['cumulative_distance']/state['n_tries'], 'n_tries=', state['n_tries'])

        # displaying the circle of TRUE GUESS
        croatia_map = cv2.circle(croatia_map, (f_inv(float(pick[2])), g_inv(float(pick[1]))), 
                           radius=7, color=(0, 0, 255), thickness=-1)
        
        cv2.imshow('guess', croatia_map)
        

if __name__ == '__main__':
    state = {'cumulative_distance':0, 'n_tries':0}
    print('q stops the game, any other key continues')
    while True:
        croatia_map = cv2.imread('assets/croatia_map.png')
        pick, imgs_guess = pick_image_and_label()

        # displaying the image
        print('pick=', pick[0])
        cv2.imshow('imgs_guess', imgs_guess)
        cv2.imshow('guess', croatia_map)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('guess', click_event, [croatia_map, pick, state])

        # wait for a key to be pressed to exit
        key = cv2.waitKey(0)
        if key==ord('q'):
            break
        print('-'*15)

    # close the window
    cv2.destroyAllWindows()


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
import msvcrt
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image
import os

out_dir =  'output'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_dir = out_dir + '/'

Trial = True
Pre_Cal = False


model = load_model('model.h5')
Q_num = 10
actl_mark = [7, 4, 1, 0, 3, 9, 8, 6, 5, 2]

#Remove the zeros
Obtained_mark = []
MagFac = 0.2
Border = 5
as_ret = 15.8/16.5 # Len/ Width
as_ret = 12/3.0

Qstn = list(range(1,(Q_num+1)))
pts_x = []
pts_y = []

pts_cnt = 0

def get_point(event,x,y,flags,param):
    global pts_cnt
    if event == cv2.EVENT_LBUTTONDOWN:
        if(pts_cnt<4):
            pts_x.append(x)
            pts_y.append(y)
            pts_cnt = pts_cnt + 1
            cv2.circle(img,(x,y),3,(0,0,0),-1)
            print('Got New Point')
        if(pts_cnt==4):
            print('Press Q to Proceed')
            


cap = cv2.VideoCapture(0)

cv2.namedWindow('Raw')

go_on = True
while(go_on):
    
    # Comment out for continuous operation
    go_on = False
    
    while(True):
        ret, img = cap.read()
        cv2.imshow('Raw',img)
        if cv2.waitKey(1) & 0xFF == ord('o'):
            break

    
    # Comment out for real case
    img = cv2.imread('input/Asol.jpg')

    img = cv2.resize(img,None, fx=MagFac,fy=MagFac)

    if(Pre_Cal== True):
        print('Select 4 points..')
        cv2.imshow('Raw',img)
        cv2.setMouseCallback('Raw',get_point)

        while(True):
            cv2.imshow('Raw',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        Points = pd.DataFrame({'X':pts_x, 'Y': pts_y})
        Points.to_csv('Perspective_Points.csv')

    else:
        Points = pd.read_csv('Perspective_Points.csv')
        pts_x = list(Points.X)
        pts_y = list(Points.Y)

    wdth = int(math.sqrt(math.pow((pts_x[0]-pts_x[1]),2) + math.pow((pts_y[0]-pts_y[1]),2)))
    pts_src = np.array([[pts_x[0], pts_y[0]], [pts_x[1], pts_y[1]], [pts_x[2], pts_y[2]], [pts_x[3], pts_y[3]]])
    pts_dts = np.array([[20, 20], [wdth, 20], [wdth, int(as_ret*wdth)],[20, int(as_ret*wdth)]])

    h, status = cv2.findHomography(pts_src, pts_dts)
    img = cv2.warpPerspective(img, h, ((wdth+50),(int(as_ret*wdth)+50)))

    cv2.imshow('Raw',img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    gray = cv2.GaussianBlur(gray,(3,3),0)
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,21,2)

    _, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]

    max_index = np.argmax(areas)
    cnt=contours[max_index]


    x,y,w,h = cv2.boundingRect(cnt)
    Table = thresh[y:y+h,x:x+w]

    if(Trial == True):
        cv2.imshow('Border', Table)
    cv2.imwrite(out_dir + 'border.jpg', Table)
        

    del areas[max_index]
    del contours[max_index]

    # Select top 2*Q_num contours

    Top_contour = []

    if(len(contours)<2*Q_num):
        print("Couldn't find all the contours... Please check the Image.")
        cv2.waitKey()
        quit()
    else:
        print("Minimum Required Contours found. Proceeding..")

    for x in range(2*Q_num):
        max_index = np.argmax(areas)
        Top_contour.append(contours[max_index])
        del areas[max_index]
        del contours[max_index]

    # Find out everyone's x and y

    X_pos = []
    Y_pos = []
    for cnt in Top_contour:
        x,y,w,h = cv2.boundingRect(cnt)
        X_pos.append(x)
        Y_pos.append(y)

    # if X value is greater than the mean, this is what we need. isolate them

    X_mean = sum(X_pos)/ len(X_pos)

    True_contour = []
    Final_Y = []

    for x in range(len(Top_contour)):
        if(X_pos[x]>X_mean):
            True_contour.append(Top_contour[x])
            Final_Y.append(Y_pos[x])

    if(len(True_contour) != Q_num):
        print("Number of Contours and Questions Do Not Match")
        cv2.waitKey()
        quit()
    else:
        print("Number of Contours = Number of Questions. Everything is okay I guess..")

    # Sort the contours based on the Y value. top ==> bottom: 1 ==> Q_num

    Final_Contour = []

    for x in range(len(True_contour)):
        max_index = np.argmax(Final_Y)
        Final_Contour.append(True_contour[max_index])
        del Final_Y[max_index]
        del True_contour[max_index]

    # In a loop, crop the contours, recognize the digits.

    thrs_dwn_lim = 150
    thrs_up = 0

    cont_num = 0
    
    for cnt in Final_Contour:
        cont_num = cont_num + 1
        
        x,y,w,h = cv2.boundingRect(cnt)
        mark = gray[y:y+h,x:x+w]
        height, width = gray.shape
        mark = mark[Border:-1*Border,Border:-1*Border]
        if(Trial== True):
            cv2.imshow('Number_Field', mark)
        cv2.imwrite(out_dir + str(cont_num) + '.jpg', mark)
        
        #Take only number
        thresh = cv2.adaptiveThreshold(mark,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,21,2)
        _, contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)

        #If use binary, uncomment the line below
        #mark = thresh

        mark = mark[y:y+h,x:x+w]
        if(w>h):
            pad = w-h
            mark = cv2.copyMakeBorder(mark,int(pad/2),(pad-int(pad/2)),0,0,0 ,0, 255)
        else:
            pad = h-w
            mark = cv2.copyMakeBorder(mark,0,0,int(pad/2),(pad-int(pad/2)),0 , 0, 255)
            
        


        mark = cv2.bitwise_not(mark)
        mark = mark+ thrs_up
        mark[mark > 255]= 255
        mark[mark < thrs_dwn_lim]= 0

        mark = cv2.resize(mark, (28,28))

        mark = mark/255.0
        if(Trial== True):
            cv2.imshow('Number Only (Resized)', mark)
            cv2.waitKey()
        
        mark = mark.reshape(-1,28,28,1)
        mark_num = model.predict(mark)
        Obtained_mark.append(np.argmax(mark_num))

    cv2.destroyAllWindows()
    print("Finished Reading")
    Obtained_mark.reverse()
    Result = pd.DataFrame({'Question':Qstn, 'Mark': Obtained_mark})
    Total = sum(Obtained_mark)
    Obtained_mark = []
    print(Result)
    print('Total: ', Total)
        

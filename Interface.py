import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


input_size = 784
hidden_sizes = [128, 64]
output_size = 10


model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9) 
criterion = nn.NLLLoss() #Negative log loss function for n classes


model.load_state_dict(torch.load('//insert pth file//'),map_location='cuda')
model.eval()
load_from_sys = True

if load_from_sys:
	hsv_value = np.load('hsv_value.npy')
# hsv_value is a 2D matrix of the hsv values saved from the images
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5), np.uint8)
canvas = None

x1 = 0
y1 = 0


noise_thresh = 800

while True:
    ret,frame = cap.read()
    if not ret:
        break
    if canvas is None:
        canvas = np.zeros_like(frame)
    canvas2 = np.zeros_like(frame)
    burred = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if load_from_sys:
        lower_range = hsv_value[0]
        upper_range = hsv_value[1]
        
    mask = cv2.inRange(frame,lower_range, upper_range)
    # eroding shrinks the objects boundaries 
    mask = cv2.erode(mask, kernel, iterations = 1)
    # dilating expands the objects boundaries
    mask = cv2.dilate(mask, kernel, iterations = 1)
    
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > 1000:
            canvas = cv2.drawContours(canvas2,contour,-1,(255,255,255),3)

    
    #frame = cv2.add(frame,canvas)
    stacked = np.hstack((canvas,frame))
    cv2.imshow('mask', stacked)
    if cv2.waitKey(1) == ord('s'):
        cv2.imwrite(os.path.join('output', 'frame.png'), stacked)
    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()
cap.release()
    

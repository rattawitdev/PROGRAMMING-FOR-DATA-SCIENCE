from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
model = load_model('65130701717_model_hand_yes_no.h5')

class_names = ['hand', 'nohand']

cap = cv2.VideoCapture(1)

_,frame_org = cap.read()
print('Frame size: ',frame_org.shape)

scale_percent = 80 # 20 # percent of original size

width = int(frame_org.shape[1] * scale_percent / 100)
height = int(frame_org.shape[0] * scale_percent / 100)
dim = (width, height)

while True:
    _,frame_org = cap.read()
    print(frame_org.shape)
    
    # resize image
    frame = cv2.resize(frame_org, dim, interpolation = cv2.INTER_AREA)
    
    im = cv2.resize(frame,(224,224))
   
    predictions = model.predict(np.array([im])) 
    
    str_class = class_names[predictions.argmax()]
        
    image = cv2.putText(frame,str_class,(50, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("frame",image)

    key = cv2.waitKey(1) #& 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

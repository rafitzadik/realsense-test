#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2
import pyrealsense as pyrs
from scipy.stats import threshold

thresh_lo = 500 # that's actually the physics of the R200 - it can detect from 50cm 
thresh_hi = 1500 #ignore anything beyond 1m for now
min_contour = 200 #area of smallest contour that we care about

pyrs.start()
dev = pyrs.Device()

cnt = 0
last = time.time()
smoothing = 0.9;
fps_smooth = 30
outFile = None

while True:

    cnt += 1
    if (cnt % 10) == 0:
        now = time.time()
        dt = now - last
        fps = 10/dt
        fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
        last = now

    dev.wait_for_frame()
    c = dev.colour
    c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
    #d = dev.depth * dev.depth_scale * 1000 #this makes d be the depth in mm
    d = dev.dac * dev.depth_scale * 1000 #this makes d be the depth in mm
    thresh = threshold(d, threshmin = thresh_lo, threshmax = thresh_hi, newval = 0) #throw out all values that are too small or too large
    thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
    thresh = thresh.astype(np.uint8)    
    #now find the contours
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rois = [] #regions of interest
    for contour in contours:
        if cv2.contourArea(contour) > min_contour:
            M = cv2.moments(contour)
            rois.append( (contour, M['m00'], (int(M['m10']/M['m00']), int(M['m01']/M['m00']))))

    #draw the ROI's on a picture
    mask = np.zeros(d.shape, np.uint8)   
    for roi in rois:
        cv2.drawContours(mask, [roi[0]], 0, 255, -1)
        #cv2.putText(pic, str(d[roi[2][1],roi[2][0]])[:4], (roi[2][0],roi[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))

    #thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    #pic = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    pic = cv2.bitwise_and(c, c, mask=mask)
    for roi in rois:
        hull = cv2.convexHull(roi[0])
        cv2.putText(pic, str(d[roi[2][1],roi[2][0]])[:4], (roi[2][0],roi[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        cv2.drawContours(pic,[hull],0,(0,0,255),2)
        
    cd = np.concatenate((c,pic), axis=1)

    cv2.putText(cd, str(fps_smooth)[:4], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

    cv2.imshow('', cd)
    if (outFile == None):
        outFile = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps_smooth, (len(cd[0]), len(cd)), True)
    if (outFile != None):
        outFile.write(cd)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

outFile.release()
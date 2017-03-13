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
min_contour = 1000 #area of smallest contour that we care about
min_palm_circle = 10

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
    pre_thresh = d
    thresh = threshold(pre_thresh, threshmin = thresh_lo, threshmax = thresh_hi, newval = 0) #throw out all values that are too small or too large
    thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
    thresh = thresh.astype(np.uint8)    
    #now dilate the image: make more things white around white areas
    kernel = np.ones((10,10), np.uint8)
    thresh_dilation = cv2.dilate(thresh, kernel, iterations=1)
    #now find the contours
    _, contours, _ = cv2.findContours(thresh_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rois = [] #regions of interest. each one is a tuple: (contour, area, center)
    #print
    for contour in contours:
        r = cv2.minAreaRect(contour)
        (r_cent, (w,h), r_angle) = r
        if cv2.contourArea(contour) > min_contour and (w / h > 1.5 or h / w > 1.5): #this just rejects squares, but ignores "mostly horizontal" or "mostly vertical". For that one would need to look both at w/h and at angle - neither is enough on its own...
            #print h, w, h/w, r_angle
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
        #get the convex hull
        hull = cv2.convexHull(roi[0])
        cv2.drawContours(pic,[hull],0,(0,0,255),2)

        #get a bounding rectangle
        rect = cv2.minAreaRect(roi[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(pic, [box], 0, (0,255,0),2)

        #get the convexity defects
        #this doesn't work on the depth map - it's too noisy, so commenting out
#        poly = cv2.approxPolyDP(roi[0], 0.001*cv2.arcLength(roi[0], True), True)
#        hull_idx = cv2.convexHull(poly, returnPoints=False)
#        defects = cv2.convexityDefects(poly, hull_idx)
#        for i in range(defects.shape[0]):
#            s,e,f,d = defects[i,0]            
#            far = tuple(poly[f][0])
#            cv2.circle(pic, far, 5, [0,0,255],-1)

        #get a line approximation
        (vx,vy,x0,y0) = cv2.fitLine(roi[0], cv2.DIST_L2, 0, 0.01, 0.01)        
        miny = min([y for [(x,y)] in roi[0]])
        maxy = max([y for [(x,y)] in roi[0]])
        minx = (miny - y0) * vx/vy + x0
        maxx = (maxy - y0) * vx/vy + x0
        cv2.line(pic, (minx,miny), (maxx, maxy), (0,0,255))

        #find the first local max enclosed circle greater than min_palm_circle
        palm_cent = None
        palm_cent_d = 0
        for y in range(int(miny), int(maxy) , 10):
            x = (y - y0) * vx/vy + x0
            diameter = cv2.pointPolygonTest(roi[0], (x,y), True)
            if (diameter > min_palm_circle and diameter > palm_cent_d):
                palm_cent = (x,y)
                palm_cent_d = diameter
            if (palm_cent != None and diameter < 0.9 * palm_cent_d):
                break #we reached a local min
        if (palm_cent != None):
            cv2.circle(pic, palm_cent, int(palm_cent_d), [0,0,255], 2)
            pixel = np.array([palm_cent[0], palm_cent[1]], np.uint)
            depth = np.array([d[pixel[1],pixel[0]]])
            #depth = np.array([d[roi[2][1],roi[2][0]]])
            #print pixel
            #print depth
            pt = dev.deproject_pixel_to_point(pixel, depth)
            #print pt
            cv2.putText(pic, '({:1.0f},{:1.0f},{:1.0f})'.format(pt[0]/10, pt[1]/10, pt[2]/10), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
            
        #print the depth of the image center
        #cv2.putText(pic, str(d[roi[2][1],roi[2][0]])[:4], (roi[2][0],roi[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
        
    cd = np.concatenate((c,pic), axis=1)

    cv2.putText(cd, str(fps_smooth)[:4], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))

    cv2.imshow('', cd)
    if (outFile == None):
        outFile = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps_smooth, (len(cd[0]), len(cd)), True)
    if (outFile != None):
        outFile.write(cd)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        cv2.waitKey(0)

outFile.release()
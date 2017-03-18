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

def detect_hand(color, depth):
    pre_thresh = depth
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
        if (cv2.contourArea(contour) > min_contour and #enough area
            (w / h > 1.5 or h / w > 1.5) and          #tall or long rect, not a square
            (min([x for [[x,y]] in contour]) > 50)):      #not at the left most part
                #print h, w, h/w, r_angle
                M = cv2.moments(contour)
                rois.append( (contour, M['m00'], (int(M['m10']/M['m00']), int(M['m01']/M['m00']))))

    if (len(rois) > 0):   
        hand = max(rois, key=lambda roi:roi[1])[0] # get the biggest roi as the hand
    else:
        return None, None, 0

    #find the first local max enclosed circle greater than min_palm_circle
    palm_cent = None
    palm_diameter = 0
    miny = min([y for [(x,y)] in hand])
    maxy = max([y for [(x,y)] in hand])
    (vx,vy,x0,y0) = cv2.fitLine(hand, cv2.DIST_L2, 0, 0.01, 0.01)        
    for y in range(int(miny), int(maxy) , 10):
        x = (y - y0) * vx/vy + x0
        diameter = cv2.pointPolygonTest(hand, (x,y), True)
        if (diameter > min_palm_circle and diameter > palm_diameter):
            palm_cent = (x,y)
            palm_diameter = diameter
        if (palm_cent != None and diameter < 0.9 * palm_diameter):
            break #we reached a local min
    return hand, palm_cent, palm_diameter

def draw_hand(color, depth, hand, palm_cent, palm_diameter, pt, pixel2, pixel3, pixel4):
        mask = np.zeros(depth.shape, np.uint8)   
        cv2.drawContours(mask, [hand], 0, 255, -1)
        pic = cv2.bitwise_and(color, color, mask=mask)
        #get the convex hull
        hull = cv2.convexHull(hand)
        cv2.drawContours(pic,[hull],0,(0,0,255),2)
        if (palm_cent != None):
            cv2.circle(pic, palm_cent, int(palm_diameter), [0,0,255], 2)
            cv2.putText(pic, '({:2.0f},{:2.0f},{:3.0f})'.format(pt[0]/10, pt[1]/10, pt[2]/10), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
#            print palm_cent
#            print pixel2
#            print pixel3
            if not np.isnan(pixel2[0]):
                cv2.line(pic, palm_cent, pixel2, (0,255,0))
            if not np.isnan(pixel3[0]):
                cv2.line(pic, palm_cent, pixel3, (0,255,0))
            if not np.isnan(pixel4[0]):
                cv2.line(pic, palm_cent, pixel4, (0,255,0))
        return pic

#def update_ball(ball_pos, ball_v, dt, palm_cent, palm_center_depth):
#    ball_pos += [a * dt for a in ball_v]
#    if (ball_pos[2] < 10):
#        ball_v[0] = random()
#        ball_v[1] = random()
#        ball_v[2] = 10
#    elif (ball_pos[2] > 3000):
#        ball_v[0] = random()
#        ball_v[1] = random()
#        ball_v[2] = -10       
#    elif (hits(ball_pos, palm_center, palm_center_depth) || ball_pos[2] > 3000):
#        ball_v = []
    
#    return ball_pos, ball_v
    
#def draw_ball(ball_pos, pic):
    
    
if __name__ == '__main__': 
    pyrs.start()
    dev = pyrs.Device()

    #Use appropriate settings here to get the exposure you need
    dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_COLOR_ENABLE_AUTO_EXPOSURE, 1)
    dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_R200_LR_AUTO_EXPOSURE_ENABLED, 1)
    #dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_COLOR_EXPOSURE, 3)
    #dev.set_device_option(pyrs.constants.rs_option.RS_OPTION_R200_LR_EXPOSURE, 3)

    cnt = 0
    last = time.time()
    smoothing = 0.9;
    fps_smooth = 30
#    outFile = None

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
        d = dev.dac * dev.depth_scale * 1000 #this makes d be the depth in mm
        hand, palm_cent, palm_diameter = detect_hand(c,d)
        if (palm_cent != None):
            pixel = np.array([palm_cent[0], palm_cent[1]], np.uint)
            depth = np.array([d[pixel[1],pixel[0]]])
            palm_cent_3d = dev.deproject_pixel_to_point(pixel, depth)
            p2 = np.array([palm_cent[0] + 10, palm_cent[1]], np.uint)
            d2 = np.array([d[p2[1],p2[0]]])
            p2_3d = dev.deproject_pixel_to_point(p2, d2)
            p3 = np.array([palm_cent[0] , palm_cent[1] + 10], np.uint)
            d3 = np.array([d[p3[1],p3[0]]])
            p3_3d = dev.deproject_pixel_to_point(p3, d3)
            p2_3d_d = p2_3d - palm_cent_3d
            p2_3d_n = p2_3d_d / np.linalg.norm(p2_3d_d) 
            p3_3d_d = p3_3d - palm_cent_3d
            p3_3d_n = p3_3d_d / np.linalg.norm(p3_3d_d) 
            p2_map = tuple(dev.project_point_to_pixel(palm_cent_3d+p2_3d_n*30))
            p3_map = tuple(dev.project_point_to_pixel(palm_cent_3d+p3_3d_n*30))
            p4_3d_d = np.array([p2_3d_n[1]*p3_3d_n[2] - p2_3d_n[2]*p3_3d_n[1], p2_3d_n[2]*p3_3d_n[0] - p2_3d_n[0]*p3_3d_n[2], p2_3d_n[0]*p3_3d_n[1] - p2_3d_n[1]*p3_3d_n[0]])
            p4_3d_n = p4_3d_d / np.linalg.norm(p4_3d_d)
            p4_map = tuple(dev.project_point_to_pixel(palm_cent_3d-p4_3d_n*30))
        else:
            palm_cent_3d = p2_map = p3_map = p4_map = None

        if (hand != None):
            pic = draw_hand(c,d,hand, palm_cent, palm_diameter, palm_cent_3d, p2_map, p3_map, p4_map)
        else:
            pic = np.zeros(c.shape, np.uint8)
    
        cd = np.concatenate((c,pic), axis=1)
    
        cv2.putText(cd, str(fps_smooth)[:4], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    
        cv2.imshow('', cd)
    #    if (outFile == None):
    #        outFile = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps_smooth, (len(cd[0]), len(cd)), True)
    #    if (outFile != None):
    #        outFile.write(cd)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            cv2.waitKey(0)

#outFile.release()
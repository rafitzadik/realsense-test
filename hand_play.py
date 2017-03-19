#!/usr/bin/env python

import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2
import pyrealsense as pyrs
from scipy.stats import threshold
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import grey_dilation, grey_erosion

thresh_lo = 700 # that's actually the physics of the R200 - it can detect from 50cm 
thresh_hi = 1200 #ignore anything beyond 1m for now
min_contour = 100 #area of smallest contour that we care about
min_palm_circle = 10
max_depth_hist = 10

def push_depth_hist(depth_hist, d):
    if (len(depth_hist) > max_depth_hist):
        depth_hist = depth_hist[:-1]
           
    #first remove small areas of zeros
    d = grey_dilation(d, size=(5,5)) 
    #and add to history
    depth_hist.insert(0, d)
    return depth_hist            

def smooth_depth(depth_hist):
    #now give the history average for each pixel:
    #return depth_hist[0]
    smooth = sum(depth_hist) / len(depth_hist)
    return smooth
    
def get_depth(d, pixel):
    return d[pixel[1],pixel[0]]
    
#nice idea, but couldn't get this to work against the backdrop of a wall that has the same hue as my hand...
def filter_hsv(color, depth, hand, palm_center, palm_radius):
    blur = cv2.blur(color, (9,9)) # remove noise
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    hue_min = 180
    hue_max = 0
    s_min = 255
    s_max = 0
    for y in range(0, int(palm_radius*0.5)): #look at 0.8 of the bottom half of the circle
        x_range = int(0.5 * np.sqrt(palm_radius * palm_radius - y * y))
        for x in range(0, x_range):
            [h, s, _] = hsv[int(palm_cent[1] + y), int(palm_cent[0]+x)]
            if (h < hue_min):
                hue_min = h
            if (h > hue_max):
                hue_max = h
            if (s < s_min):
                s_min = s
            if (s > s_max):
                s_max = s
    #now mask HSV to only our dilated hand and filter on that Hue
    hue_min = int(hue_min * 0.8)
    hue_max = int(hue_max * 1.2)
    s_min = int(s_min * 0.8)
    s_max = int(s_max * 1.2)
    print hue_min, hue_max, s_min, s_max        
    mask = np.zeros(depth.shape, np.uint8)            
    cv2.drawContours(mask, [hand], 0, 255, -1)
    hsv_hand = cv2.bitwise_and(hsv, hsv, mask=mask)
    hsv_min = np.array([hue_min, s_min, 5], np.uint8)
    hsv_max = np.array([hue_max, s_max, 250], np.uint8)
    hsv_thresh = cv2.inRange(hsv_hand, hsv_min, hsv_max)
    _, hsv_contours, _ = cv2.findContours(hsv_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print 'depth based area: ' + str(cv2.contourArea(hand))
    if (len(hsv_contours) > 0):
        hand = max(hsv_contours, key=lambda cnt:cv2.contourArea(cnt))
        print 'hsv based area: ' + str(cv2.contourArea(hand))
    return hand

def my_canny(image, sigma=0.7):
    lower = 10
    upper = 200
    edged = cv2.Canny(image, lower, upper)
    return edged

def detect_hand(color, depth):
    thresh = threshold(depth, threshmin = thresh_lo, threshmax = thresh_hi, newval = 0) #throw out all values that are too small or too large
    thresh = threshold(thresh, threshmax = 1, newval = 255) #make remaining values 255
    thresh = thresh.astype(np.uint8)    
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    thresh_dilation = cv2.dilate(thresh, kernel, iterations=1)

    _, contours, _ = cv2.findContours(thresh_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    rois = [] #regions of interest. each one is a tuple: (contour, area, center)
    #print
    print 'len(contours): ', len(contours)
    for contour in contours:
        r = cv2.minAreaRect(contour)
        (r_cent, (w,h), r_angle) = r
        if cv2.contourArea(contour) > min_contour and min([x for [[x,y]] in contour]) > 50:
                M = cv2.moments(contour)
                rois.append( (contour, M['m00'], (int(M['m10']/M['m00']), int(M['m01']/M['m00']))))

    if (len(rois) > 0):   
        hand = max(rois, key=lambda roi:roi[1])[0] # get the biggest roi as the hand
    else:
        return None, None, 0

    #find the first local max enclosed circle greater than min_palm_circle
    palm_cent = None
    palm_radius = 0
    miny = min([y for [(x,y)] in hand])
    maxy = max([y for [(x,y)] in hand])
    (vx,vy,x0,y0) = cv2.fitLine(hand, cv2.DIST_L2, 0, 0.01, 0.01)        
    for y in range(int(miny), int(maxy) , 10):
        x = float((y - y0) * vx/vy + x0)
        radius = cv2.pointPolygonTest(hand, (x,y), True)
        if (radius > min_palm_circle and radius > palm_radius):
            palm_cent = (int(x),int(y))
            palm_radius = radius
        if (palm_cent != None and radius < 0.9 * palm_radius):
            #hand = filter_hsv(color, depth, hand, palm_cent, palm_radius)    
            break #we reached a local min
    return hand, palm_cent, palm_radius
    
def color_to_canny(pic):
    gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = my_canny(blurred)
    return cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

def palm_vectors(dev, d, palm_cent):
    p2 = np.array([palm_cent[0] + 10, palm_cent[1]], np.uint)
    #d2 = np.array([d[p2[1],p2[0]]])
    d2 = get_depth(d, p2)
    p2_3d = dev.deproject_pixel_to_point(p2, d2)
    p3 = np.array([palm_cent[0] , palm_cent[1] + 10], np.uint)
    #d3 = np.array([d[p3[1],p3[0]]])
    d3 = get_depth(d, p3)
    p3_3d = dev.deproject_pixel_to_point(p3, d3)
    p2_3d_d = p2_3d - palm_cent_3d
    p2_3d_n = p2_3d_d / np.linalg.norm(p2_3d_d) 
    p3_3d_d = p3_3d - palm_cent_3d
    p3_3d_n = p3_3d_d / np.linalg.norm(p3_3d_d) 
    p2_map = tuple(dev.project_point_to_pixel(palm_cent_3d+p2_3d_n*30))
    p3_map = tuple(dev.project_point_to_pixel(palm_cent_3d+p3_3d_n*30))
    p4_3d_d = np.array([p2_3d_n[1]*p3_3d_n[2] - p2_3d_n[2]*p3_3d_n[1], p2_3d_n[2]*p3_3d_n[0] - p2_3d_n[0]*p3_3d_n[2], p2_3d_n[0]*p3_3d_n[1] - p2_3d_n[1]*p3_3d_n[0]])
    p4_3d_n = p4_3d_d / np.linalg.norm(p4_3d_d)
    p4_map = tuple(dev.project_point_to_pixel(palm_cent_3d-p4_3d_n*60))
    return p2_map, p3_map, p4_map
    
def draw_hand(color, depth, hand, palm_cent, palm_diameter, pt, pixel2, pixel3, pixel4):
    mask = np.zeros(depth.shape, np.uint8)   
    cv2.drawContours(mask, [hand], 0, 255, -1)
    pic = cv2.bitwise_and(color, color, mask=mask)
    #pic = cv2.bitwise_or(pic, color_to_canny(pic))
    #get the convex hull
    #hull = cv2.convexHull(hand)
    #cv2.drawContours(pic,[hull],0,(0,0,255),2)
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
    
def smooth(c, r, hist):
    max_hist = 10
    if (len(hist) > max_hist):
        hist = hist[:-1]
    if c == None:
        cx = None
        cy = None
    else:
        cx = c[0]
        cy = c[1]
    hist.insert(0, (cx, cy ,r))
    c_sum_x = 0
    c_sum_y = 0
    r_sum = 0
    n = 0
    for elm in hist:
        if (elm[0] != None):
            c_sum_x += elm[0]
            c_sum_y += elm[1]
            r_sum += elm[2]
            n += 1
    if n > 0:
        return (c_sum_x/n, c_sum_y/n), r_sum/n, hist
    else:
        return None, None, hist
    
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
    hist = []
    depth_hist = []
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
        depth_hist = push_depth_hist(depth_hist, d)
        d = smooth_depth(depth_hist)
        hand, palm_cent, palm_radius = detect_hand(c,d)
        palm_cent_smooth, palm_radius_smooth, hist = smooth(palm_cent, palm_radius, hist)
        if (palm_cent_smooth != None):
            pixel = np.array([palm_cent_smooth[0], palm_cent_smooth[1]], np.uint)
            #depth = np.array([d[pixel[1],pixel[0]]])
            depth = get_depth(d, pixel)
            palm_cent_3d = dev.deproject_pixel_to_point(pixel, depth)
            p2_map, p3_map, p4_map = palm_vectors(dev, d, palm_cent_smooth)
        else:
            palm_cent_3d = p2_map = p3_map = p4_map = None

        if (hand != None):
            pic = draw_hand(c,d,hand, palm_cent_smooth, palm_radius_smooth, palm_cent_3d, p2_map, p3_map, p4_map)
        else:
            pic = np.zeros(c.shape, np.uint8)
        
        #pic = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_RAINBOW)    
        cd = np.concatenate((c,pic), axis=1)
    
        #cv2.putText(cd, str(fps_smooth)[:4], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
        cv2.putText(cd, str(d[240, 320])[:4], (0,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255))
    
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
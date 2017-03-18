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
min_contour = 5000 #area of smallest contour that we care about
min_palm_circle = 10

def avg_depth(depth, pt):
#find the depth at pont (x,y) by averaging out the meaningful values around (x,y)
    max_hist = 10
    try:
        if (len(avg_depth.hist) > max_hist):
            avg_depth.hist = avg_depth.hist[:-1]
    except AttributeError:
        avg_depth.hist = []
    avg_depth.hist.insert(0, depth)
    
    [x,y] = pt 
    x = int(x)
    y = int(y)
    if (x < 2): x = 2
    if (x > len(depth[0]) - 2): x = len(depth[0]-2)
    if (y < 2): y = 2
    if (y > len(depth) - 2): y = len(depth)-2
    #return depth[y,x]
        
    d = 0
    n = 0
    for h in avg_depth.hist:
        for i in range(y-2, y+3):
            for j in range(x-2, x+3):
                if depth[i,j] > 10:
                    n += 1
                    d += h[i,j]
    if n > 0:            
        return np.array([(d / n)])
    else:
        return np.array([0])
    
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
    d2 = avg_depth(d, p2)
    p2_3d = dev.deproject_pixel_to_point(p2, d2)
    p3 = np.array([palm_cent[0] , palm_cent[1] + 10], np.uint)
    #d3 = np.array([d[p3[1],p3[0]]])
    d3 = avg_depth(d, p3)
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
        hand, palm_cent, palm_radius = detect_hand(c,d)
        palm_cent_smooth, palm_radius_smooth, hist = smooth(palm_cent, palm_radius, hist)
        if (palm_cent_smooth != None):
            pixel = np.array([palm_cent_smooth[0], palm_cent_smooth[1]], np.uint)
            #depth = np.array([d[pixel[1],pixel[0]]])
            depth = avg_depth(d, pixel)
            palm_cent_3d = dev.deproject_pixel_to_point(pixel, depth)
            p2_map, p3_map, p4_map = palm_vectors(dev, d, palm_cent_smooth)
        else:
            palm_cent_3d = p2_map = p3_map = p4_map = None

        if (hand != None):
            pic = draw_hand(c,d,hand, palm_cent_smooth, palm_radius_smooth, palm_cent_3d, p2_map, p3_map, p4_map)
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
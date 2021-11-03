import numpy as np
import cv2 as cv

def hcn (contours):
    result =[]  
    for i in contours:
        area = cv.contourArea(i)
        if area>2000:
            # tính chu vi và sử dụng contours approximation algorithm để đơn giản hóa contours
            D = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,D*0.01,True)
            if len(approx) == 4:
                result.append(approx)
    result = sorted(result, key=cv.contourArea, reverse=True)
    return result

def order(points):

    return_points = np.zeros_like(points,dtype=np.int32)
    points = points.reshape((4,2))
    
    order_00_11 = np.sum(points,axis=1)
    order_01_10 = map(lambda x: x[0]-x[1], points)
    order_01_10 = [i for i in order_01_10]
    return_points[0] = points[np.argmin(order_00_11)]
    return_points[1] = points[np.argmax(order_01_10)]
    return_points[2] = points[np.argmin(order_01_10)]
    return_points[3] = points[np.argmax(order_00_11)]

    return return_points  #2d array


def checkans(imgThresh, total_questions:int):
    rs = imgThresh.shape[1]//total_questions
    imgThresh = cv.resize(imgThresh,(int(total_questions*rs),int(total_questions*rs)))
    # print(len(imgthresh))
    hor = np.vsplit(imgThresh,total_questions)
    store = []
    for i in hor:
        ver = np.hsplit(i,4)
        
        each_question = []
        for j in ver:
            each_question.append(cv.countNonZero(j))
        store.append(each_question)
    store_ndarray = np.array(store)
    print(store_ndarray)
    
    result = map(lambda x: np.where(x==np.max(x)),store_ndarray) # map object
    result = [i[0][0] for i in result] # list object
    # print(result)

    return result

def drawAns (img,this_img_answers:list,true_answers:list):
    x = img.shape[0]//4
    y = img.shape[1]//len(this_img_answers)
    
    for i in range(len(this_img_answers)):
        ans = true_answers[i] #[0,1,3,2,0,1,2]
        # print(ans)

        pos_x = (ans * x) + x//2
        pos_y = (i * y) + y//2
        if ans != this_img_answers[i]:
            this_ans = this_img_answers[i]
            w_pos_x = (this_ans * x) + x//2 
            w_pos_y = (i * y) + y//2
            cv.circle(img,(w_pos_x,w_pos_y),5,(0,0,255),10)
        cv.circle(img,(pos_x,pos_y),5,(0,255,0),5)
    return img
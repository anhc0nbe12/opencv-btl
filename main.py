import cv2 as cv
import numpy as np
import funcs


img = cv.imread('anhdethi.jpg')
img = cv.resize(img,(700,700))
total_questions= 7

answer = [0,1,3,2,1,1,1]
imgCopy_1 = img.copy()
imgCopy_2 = img.copy()
imgCopy_3 = img.copy()
# xử lý độ sáng của ảnh
dilated_img = cv.dilate(img, np.ones((7,7), np.uint8)) 
new_img = 255 - cv.absdiff(img, dilated_img) # lấy nền trắng 

#tìm đường biên
imgGray = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(7,7),1,1)
imgCanny = cv.Canny(imgBlur,20,50)



# tìm các contours trong ảnh
contours,hierarchy = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgCopy_1,contours,-1,(0,255,0),1)

# tìm ô khoanh đáp án và lấy 4 góc của nó
biggest_contour = funcs.hcn(contours)[0] 
# print(len(biggest_contour))
# print(biggest_contour)

# vẽ 4 góc của ô khoanh đáp án lên ảnh
cv.drawContours(imgCopy_2, biggest_contour,-1,(255,0,0),5)

# thiết lập lại vị trí của 4 điểm trên ô khoanh
biggest_contour =  funcs.order(biggest_contour)

# cắt ảnh
pt1 = np.float32(biggest_contour)
pt2 = np.float32([[0,0],[700,0],[0,700],[700,700]])
matrix = cv.getPerspectiveTransform(pt1,pt2)
imgBoard = cv.warpPerspective(new_img,matrix,(700,700))

# binary img
imgBoardGray = cv.cvtColor(imgBoard,cv.COLOR_BGR2GRAY)
imgThresh = cv.threshold(imgBoardGray,245,255,cv.THRESH_BINARY_INV)[1]
# cv.imshow('thresh1', imgThresh)



# tính toán kết quả 
result = funcs.checkans(imgThresh,total_questions)
score = 0
for i in range(total_questions):
    if result[i] == answer[i]:
        score+=1
# print(score)



# đáp án đã khoanh và kết quả
mark_img = funcs.drawAns(imgBoard,result,answer)
black_board = np.zeros_like(imgBoard)
black_board = funcs.drawAns(black_board,result,answer)

inv_matrix = cv.getPerspectiveTransform(pt2,pt1)
inv_imgBoard = cv.warpPerspective(black_board,inv_matrix,(700,700))

# print(imgCopy_3.shape, inv_imgBoard.shape)
imgCopy_3 = cv.addWeighted(imgCopy_3,1,inv_imgBoard,1,0)

# output
cv.imshow('all answers',mark_img)
cv.imshow('black',black_board)
cv.imshow('inv_board',inv_imgBoard)
cv.imshow('final',imgCopy_3)
cv.imshow('equa',imgBoardGray)
cv.imshow('thresh', imgThresh)
# funcs.findContours(imgCanny)

# cv.imshow('anh goc: ',img)
# cv.imshow('anh gray: ',imgGray)
# cv.imshow('bang', imgBoard)
# cv.imshow('anh thresh: ',imgCanny)


cv.waitKey(0)
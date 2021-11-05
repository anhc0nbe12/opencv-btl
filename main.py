import cv2 as cv
import numpy as np
import funcs
import matplotlib.pyplot as plt

img = cv.imread('anhdethi3.jpg')
w,h = 700,700
img = cv.resize(img,(w,h))
# total_questions = 7
# answer = [0,1,3,2,1,1,1]
total_questions = 8
answer = [0,1,3,2,1,1,1,3]

imgCopy_1 = img.copy()
imgCopy_2 = img.copy()
imgCopy_3 = img.copy()
# xử lý độ sáng của ảnh
dilated_img = cv.dilate(img, np.ones((5,5), np.uint8)) 
new_img = 255 - cv.absdiff(img, dilated_img) # lấy nền trắng 
#tìm đường biên
imgGray = cv.cvtColor(new_img,cv.COLOR_BGR2GRAY)
imgBlur = cv.GaussianBlur(imgGray,(5,5),1)
imgCanny = cv.Canny(imgBlur,20,50)


# tìm các contours trong ảnh
contours,_ = cv.findContours(imgCanny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgCopy_1,contours,-1,(0,255,0),1)

# tìm ô khoanh đáp án và lấy 4 góc của nó
biggest_contour = funcs.hcn(contours)[0] 

# vẽ 4 góc của ô khoanh đáp án lên ảnh
cv.drawContours(imgCopy_2, biggest_contour,-1,(255,0,0),5)

# thiết lập lại vị trí của 4 điểm trên ô khoanh
biggest_contour =  funcs.order(biggest_contour)

# cắt ảnh
pt1 = np.float32(biggest_contour)
pt2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
matrix = cv.getPerspectiveTransform(pt1,pt2)
imgBoard = cv.warpPerspective(new_img,matrix,(w,h))

# binary img
imgBoardGray = cv.cvtColor(imgBoard,cv.COLOR_BGR2GRAY)
imgThresh = cv.threshold(imgBoardGray,240,255,cv.THRESH_BINARY_INV)[1]
imgThresh = cv.medianBlur(imgThresh,5)
# cv.imshow('thresh1', imgThresh)


# tính toán kết quả 
funcs.checkans(imgThresh,total_questions)
result = funcs.checkans(imgThresh,total_questions)
score = 0
dict = {
    0:'A',1:'B',2:'C',3:'D'
}
for idx, i in enumerate(result):
    out = f"Câu {idx} khoanh: "+ dict[i]
    score +=1
    if answer[idx] != i:
        out += ', đáp án chính xác: '+ dict[answer[idx]]
        score -=1
    print(out)
print("Điểm số: ",score,'/',len(answer))





# đáp án đã khoanh và kết quả
mark_img = funcs.drawAns(imgBoard,result,answer)
black_board = np.zeros_like(imgBoard)
black_board = funcs.drawAns(black_board,result,answer)

inv_matrix = cv.getPerspectiveTransform(pt2,pt1)
inv_imgBoard = cv.warpPerspective(black_board,inv_matrix,(w,h))

# print(imgCopy_3.shape, inv_imgBoard.shape)
imgCopy_3 = cv.addWeighted(imgCopy_3,1,inv_imgBoard,1,0)

# output
# cv.imshow('all answers',mark_img)
# cv.imshow('black',black_board)
# cv.imshow('inv_board',inv_imgBoard)
# cv.imshow('final',imgCopy_3)
# cv.imshow('equa',imgBoardGray)
# cv.imshow('thresh', imgThresh)

img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
imgEqual = cv.cvtColor(imgBoardGray,cv.COLOR_BGR2RGB)
imgCanny = cv.cvtColor(imgCanny,cv.COLOR_BGR2RGB)
imgCopy_1 = cv.cvtColor(imgCopy_1,cv.COLOR_BGR2RGB)
imgThresh = cv.cvtColor(imgThresh,cv.COLOR_BGR2RGB)
imgFinal = cv.cvtColor(imgCopy_3,cv.COLOR_BGR2RGB)

_,index = plt.subplots(2,3)
index[0,0].imshow(img),index[0,0].set_title('anh goc')
index[0,1].imshow(new_img),index[0,1].set_title('can bang do sang ')
index[0,2].imshow(imgCanny),index[0,2].set_title('Canny ')
index[1,0].imshow(imgCopy_1),index[1,0].set_title('tim contours ')

index[1,1].imshow(imgThresh),index[1,1].set_title('Thresh ')
index[1,2].imshow(imgFinal),index[1,2].set_title('Ket qua ')



plt.show()


cv.waitKey(0)
cv.destroyAllWindows()
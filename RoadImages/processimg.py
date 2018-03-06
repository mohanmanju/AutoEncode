import cv2

img = cv2.imread("xyz.jpg")
# img = cv2.resize(img,(800,800))
# cv2.imwrite("xyz.jpg",img);
cv2.imshow("dfdf",img)
cv2.waitKey(0)

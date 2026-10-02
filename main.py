import cv2
image = cv2.imread("images/IMG_20261002_201009.jpg")
cv2.imshow("MiniVision", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
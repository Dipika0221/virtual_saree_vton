import cv2

# load cloth image
img = cv2.imread("HR-VITON/HR-VITON-main/datasets/test/cloth/saree.jpg")

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# remove white background
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# save mask
cv2.imwrite("HR-VITON/HR-VITON-main/datasets/test/cloth-mask/saree.jpg", mask)

cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
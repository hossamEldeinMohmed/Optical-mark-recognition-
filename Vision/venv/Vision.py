#حسام الدين محمد عثمان
#حسام عبد الحكيم سعيد عرابي
#cs1
#عدد الاسالة 5
#الخيارات المتاحة في كل سوال  ,a,b,c,d,e

import cv2
import numpy as np
import cv2 as cv
import tkinter
import _tkinter
import imutils
from  imutils.perspective import four_point_transform
from imutils import contours
#def anser()
def read(loc):
  img = cv2.imread(loc)
  cv2.imshow('image', img)
  cv2.waitKey(0)
  return img

def step_1(img):
   edged =cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(5,5),0),90,185,5)
   cv2.imshow('image',edged)
   cv2.waitKey(0)
   return edged


def step_2(edged,img):
   cnts = cv2.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
   for c in cnts:
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)
      if len(approx) == 4:
         docCnt = approx
         break

   ANSWER_KEY = {0: int(x2), 1: int(y2), 2: int(z2), 3: int(d2), 4: int(w2)}
   paper = four_point_transform(img, docCnt.reshape(4, 2))
   warped = four_point_transform(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), docCnt.reshape(4, 2))
   ink=cv.drawContours(img, cnts, 0, (0, 255, 0), 3)

   cv2.imshow('image',ink )
   cv2.waitKey(0)
   thresh = cv2.threshold(warped, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
   cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                           cv2.CHAIN_APPROX_SIMPLE)
   cnts = imutils.grab_contours(cnts)
   questionCnts = []
   for c in cnts:
      (x, y, w, h) = cv2.boundingRect(c)
      ar = w / float(h)
      if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
         questionCnts.append(c)
   questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
   correct = 0

   for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
      cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
      bubbled = None
      for (j, c) in enumerate(cnts):

         mask = np.zeros(thresh.shape, dtype="uint8")
         cv2.drawContours(mask, [c], -1, 255, -1)
         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
         total = cv2.countNonZero(mask)


         if bubbled is None or total > bubbled[0]:
           bubbled = (total, j)
      color = (0, 0, 255)
      k = ANSWER_KEY[q]
      if k == bubbled[1]:
         color = (0, 255, 0)
         correct += 1
      cv2.drawContours(paper, [cnts[k]], -1, color, 3)
   score = (correct / 5.0) * 100
   print("[INFO] score: {:.2f}%".format(score))
   cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
   cv2.imshow("Original", img)
   cv2.imshow("Exam", paper)
   cv2.waitKey(0)
   return
print("enter the image path with // instead of / ")
s=input("like C:\\\\Users\\\\Administrator\\\\Documents\\\\MATLAB\\\\123.jpg \n");


print("enter 0 for a ,1 for b,2 for c,3 for d,4 for e")
x2=input("QuestionAnser #1: \n")
y2=input("QuestionAnser #2: \n")
z2=input("QuestionAnser #3: \n")
d2=input("QuestionAnser #4: \n")
w2=input("QuestionAnser #5: \n")
im=read(s)
e=step_1(im);
cv2.waitKey(0)
step_2(e,im);



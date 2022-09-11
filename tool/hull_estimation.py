# import cv2
# import torch
# import numpy as np
#
# # 读取图片并转至灰度模式
# imagepath = 'picture_16.png'
# img = cv2.imread(imagepath, 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # _, bunary_img = cv2.threshold(gray, 170, 1, cv2.THRESH_BINARY) # modify the gray image to binary one
#
# contours, hierarchy = cv2.findContours(gray, 0, 1) # contour of the objects in an image, gray is an binary image
#
# regions = []
# for c in contours:
#     regions.append(cv2.contourArea(c))
# cnt_max = contours[np.argsort(-np.array(regions))[0]]
# max_id = np.argsort(-np.array(regions))[0]
#
# hull = cv2.convexHull(cnt_max)
#
# # length = len(contours[0])
# # for i in range(len(contours[0])):
# #     cv2.line(img, tuple(contours[0][i][0]), tuple(contours[0][(i+1)%length][0]), (0,255,0), 3)
# # cv2.imshow('contour', img)
# # cv2.waitKey(100)
# # cv2.destroyAllWindows()
#
# length = len(hull)
# for i in range(len(hull)):
#     cv2.line(img, tuple(hull[i][0]), tuple(hull[(i+1)%length][0]), (0,255,0), 2)
# cv2.imshow('hull', img)
# cv2.waitKey(100)
# cv2.destroyAllWindows()
#
#
# tmp1 = (torch.zeros_like(torch.Tensor(gray))).numpy()
# cv2.fillConvexPoly(tmp1, hull, (255,0,255))
#
# cv2.imshow('hull_fill', tmp1) # show the convex hull on the image
# cv2.waitKey(100)
# cv2.destroyAllWindows()
# # compute the region inside of the convex hull
#
#
#
# # give value according to the indexes
# a = torch.zeros([5,5])
# index = (torch.LongTensor([0,1]), torch.LongTensor([1,2]))
# value = torch.Tensor([1,1])
# a.index_put_(index, value)
# print(a)
from PIL import Image
from torchvision import transforms

from tool.showImag import multi_slice_viewer_debug

totensor= transforms.ToTensor()
img = Image.open('picture_18.png')
gt = Image.open('picture_18_gt.png')

img = totensor(img)
gt = totensor(gt)
multi_slice_viewer_debug([img], cmap='Blues')

multi_slice_viewer_debug([img], gt, no_contour=True)


from scipy.ndimage import zoom
from torchvision import transforms
from PIL import Image
from cv2 import cv2
import cv2
import numpy as np
import torch


def gaussian_filter(img, K_size=7, sigma=0.4):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy()
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W]
    return out


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


if __name__ == '__main__':

    trans = transforms.ToTensor()
    pp = np.loadtxt('model2_img1.txt')
    ct_img = cv2.imread('1.png', 1)

    img1 = cv2.resize(pp, (224, 224))
    img2 = np.float32(cv2.resize(ct_img, (224, 224))) / 255
    ct = img2
    img1 = gaussian_filter(img1).transpose(2,0,1)[0]


    means = [0.5]
    stds = [1]
    preprocessed_img = img1.copy()[:, :]
    preprocessed_img[:,:] = preprocessed_img[:, :] - means
    preprocessed_img[:,:] = preprocessed_img[:, :] / stds
    preprocessed_img = np.ascontiguousarray(preprocessed_img)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    input1 = preprocessed_img
    heatmap = cv2.applyColorMap(np.uint8(255 * input1), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    heatmap = torch.where(torch.from_numpy(heatmap) < 0.6, torch.tensor([0.0]), torch.from_numpy(heatmap))
    cam = heatmap + np.float32(img2)
    cam = cam / cam.max()

    cv2.imwrite("cam.jpg", np.uint8(255 * cam))

    cam = torch.where(torch.from_numpy(heatmap) < 0.005, torch.from_numppy(img2), torch.from_numpy(heatmap+np.float32(img2)))
import pydicom as dicom
import matplotlib.pylab as plt
import os

# specify your image path
image_path = '../dataset/teeth'
img_list = os.listdir(image_path)
for img in img_list:
    ds = dicom.dcmread(f"{image_path}/{img}")
    pixel_array_numpy = ds.pixel_array
    plt.imshow(ds.pixel_array)




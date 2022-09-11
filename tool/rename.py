import os


strs = '../dataset/Spleen'

# img = f'{strs}/imagesTr'
# gt = f'{strs}/labeldsTr'

# for file_png in os.listdir(img):
#     if int(file_png[7:-7]) < 10:
#         os.rename(f'{img}/{file_png}', f'{img}/{file_png[:7]}0{file_png[7:]}')

# for file_png in os.listdir(gt):
#     if int(file_png[7:-7]) < 10:
#         os.rename(f'{gt}/{file_png}', f'{gt}/{file_png[:7]}0{file_png[7:]}')


train_gt = f'{strs}/train/gt'
train_img = f'{strs}/train/img'
val_gt = f'{strs}/val/gt'
val_img = f'{strs}/val/img'
for file_png in os.listdir(train_gt):
    if int(file_png[10:-4]) < 10:
        os.rename(f'{train_gt}/{file_png}', f'{train_gt}/{file_png[:10]}0{file_png[10:]}')

for file_png in os.listdir(train_img):
    if int(file_png[10:-4]) < 10:
        os.rename(f'{train_img}/{file_png}', f'{train_img}/{file_png[:10]}0{file_png[10:]}')

for file_png in os.listdir(val_gt):
    if int(file_png[10:-4]) < 10:
        os.rename(f'{val_gt}/{file_png}', f'{val_gt}/{file_png[:10]}0{file_png[10:]}')

for file_png in os.listdir(val_img):
    if int(file_png[10:-4]) < 10:
        os.rename(f'{val_img}/{file_png}', f'{val_img}/{file_png[:10]}0{file_png[10:]}')

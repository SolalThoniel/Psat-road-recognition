import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageOps


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def crop_resize(width, height, file_path):
    for file in os.listdir(file_path):
        f_img = file_path + "/" + file
        img = Image.open(f_img)
        img = ImageOps.fit(img, [width, height], method=0, bleed=0.0, centering=(0.5, 0.5))
        new_file_path = file_path + '/thumbnail/' + file.split('.')[0] + '_MIT_copy.' + file.split('.')[1]
        img.save(new_file_path)


if __name__ == '__main__':
    print('utils')

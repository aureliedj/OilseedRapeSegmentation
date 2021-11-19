from glob import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Get Binary
def GetBinary(msk):
    msk_arr = np.array(msk)
    msk_arr[msk_arr > 127] = 255
    msk_arr[msk_arr <= 127] = 0
    new_msk = Image.fromarray(msk_arr)
    return new_msk

def rescale1400(image):
    width, height = image.size
    if width >= height:
        angle = 90
        out = image.rotate(angle, expand=True)
    else:
        out = image
    W = 1120
    H = 1880
    resized = out.resize((W, H), Image.NEAREST)
    return resized

def split4(im):

    # Setting the points for cropped image
    output = []
    for i in np.arange(4):
        for j in np.arange(4):
            img = im.crop((i*280, j*470, i*280+280, j*470+470)) #280, 470
            output.append(img)
    return output


if __name__ == "__main__":

    image_path = r"E:\Todo\cropped"
    #mask_path = r"H:\Aurélie\Data\ModelB\Train\Join"
    outpath_im = r"C:\Users\A60026184\Desktop\ModelA\Todo"
    #outpath_mask = r"\Masks"

    image_files = glob(image_path + '\\*.*')

    for f in image_files:

        filename = f.split('\\')[-1]#[8:] #CROP MODEL A

        #mask_file = glob(mask_path + '\\*' + filename)
        #if mask_file == []:
        #    print(filename, 'is missing.')
        #    continue

        img = Image.open(f)
        #msk = Image.open(mask_file[0])
        #msk = GetBinary(msk)

        new_img = rescale1400(img)
        #new_msk = rescale1400(msk)

        splitted_im = split4(new_img)
        #splitted_mk = split4(new_msk)

        for i, split_im in enumerate(splitted_im):
            if i < 10:
                num = '0'+str(i)
            else:
                num = str(i)
            split_im.save(outpath_im + "\\" + num + '_' + filename)
        #for i, split_mk in enumerate(splitted_mk):
        #    split_mk.save(outpath_mask + "\\" + str(i) + '_' + filename)
        print(filename, 'split and saved.')


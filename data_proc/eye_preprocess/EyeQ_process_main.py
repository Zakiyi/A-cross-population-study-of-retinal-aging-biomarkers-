import fundus_prep as prep
from glob import glob
import os
import sys
sys.path.append(os.getcwd())
import cv2 as cv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tqdm import tqdm


def process(image_list, save_path):
    
    for image_path in tqdm(image_list):
        dst_image = os.path.split(image_path)[-1].replace('.jpg','.png')
        dst_path = os.path.join(save_path, dst_image)
        if os.path.exists(dst_path):
            print('continue...')
            continue

        img = prep.imread(image_path)
        try:
            r_img, borders, mask, _,_ = prep.process_without_gb(img)
            r_img = cv.resize(r_img, (800, 800))
            prep.imwrite(dst_path, r_img)
            # mask = cv.resize(mask, (800, 800))
            # prep.imwrite(os.path.join('./original_mask', dst_image), mask)
        except Exception as e:
            r_img = cv.resize(img, (800, 800))
            prep.imwrite(dst_path, r_img)
            print(image_path)
            print(e)
            # break
            continue


if __name__ == "__main__":
    #image_list = glob(os.path.join('/media/zyi/litao/retinal_age_projects/data/Dunnedine study/images', '*.JPG'))
    import pandas as pd
    csv_file = pd.read_csv('data/ukb-1/ukb_repeated.csv')
    image_list = ['data/ukb-1' + img for img in csv_file.filename.array]
    save_path = prep.fold_dir('data/ukb-1')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    process(image_list, save_path)

        






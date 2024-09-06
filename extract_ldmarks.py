import os, cv2, glob
import numpy as np

from os.path import join, basename, dirname
from tqdm import tqdm
from process_slpt_landmarks import SLPTWrapper_singleframe


SLPT = SLPTWrapper_singleframe(modelpath='process_slpt_landmarks/weights/WFLW_6_layer.pth')

def get_filelist(dataroot, filelist_path):
    """Return a list of image paths."""
    filelists = []
    with open(filelist_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if'' in line: line = line.split()[0]
            filelists.append(join(dataroot, line))
            
    return filelists

if __name__ == '__main__':
    dataroot = '/data/test-db/home/liyongyuan/Datasets/FFHQ'
    filelist_path = '/data/test-db/home/liyongyuan/Datasets/filelists/FFHQ.txt'

    filelist = get_filelist(dataroot, filelist_path)

    img_dir = '/data/test-db/home/liyongyuan/Portrait-4D/asserts/test_avatar/zhibo_batchtest/lijiaqi_cut/ori_images'
    filelist = glob.glob(join(img_dir, '*.png'))
    
    for img_path in tqdm(filelist, desc='Processing images: extracting landmarks'):
        img = cv2.imread(img_path)
        
        lmark_path = img_path.replace('ori_images', '2dldmks_align').replace('.png', '.npy')
        lmark = SLPT.get_lmark(img)
        
        np.save(lmark_path, lmark)
    
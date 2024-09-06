import os, argparse, time
import torch, cv2, traceback

import numpy as np
import torch.nn as nn

from os.path import join, basename, dirname, splitext
from PIL import Image
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf

import sys
# sys.path.append(join(os.getcwd(), '..'))
sys.path.append(os.getcwd())

from lib.face_detect_ldmk_pipeline import FaceLdmkDetector
from lib.crop_images_portrait_model import align_img_bfm
from lib.models.facerecon.facerecon import FaceReconModel
from lib.models.fd.fd import faceDetector
from lib.models.ldmk.ldmk import ldmkDetector
from lib.models.ldmk.ldmk import ldmk3dDetector


class PreprocessImg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda')
        self.cfg = OmegaConf.load(self.args.config)

        self.net_recon = FaceReconModel(self.cfg.model.facerecon)
        self.net_fd = faceDetector(self.cfg)
        self.net_ldmk = ldmkDetector(self.cfg)
        self.net_ldmk_3d = ldmk3dDetector(self.cfg)

        self.net_recon.device = torch.device('cuda')  # net_recon does not inherit torch.nn.module class
        self.net_recon.net_recon.cuda()
        self.net_fd.cuda()
        self.net_ldmk.cuda()
        self.net_ldmk_3d.cuda()

        ### set eval and freeze models
        self.net_recon.eval()
        self.net_fd.eval()
        self.net_ldmk.eval()
        self.net_ldmk_3d.eval()
    
        self.fd_ldmk_detector = FaceLdmkDetector(self.net_fd, self.net_ldmk, self.net_ldmk_3d)
        self.stand_index = np.array([96, 97, 54, 76, 82])
        
        self.target_size = 512. # target image size after cropping
        self.rescale_factor = 180 # a scaling factor determining the size of the head in the image, ffhq crop size seems as eg3d crop
        

    def _crop_extr_img(self, input_dir, save_dir, video=True, use_crop_smooth=False, process_frames=99999):
        test_start = time.time()
        
        # align_dir = os.path.join(save_dir, "align_images")
        # save_dir_bfm = os.path.join(save_dir, "bfm_params")
        # save_dir_bfm_vis = os.path.join(save_dir, "bfm_vis")
        # save_dir_ldmk2d = os.path.join(save_dir, "2dldmks_align")
        # save_dir_ldmk3d = os.path.join(save_dir, "3dldmks_align")
        # os.makedirs(align_dir, exist_ok=True)
        # os.makedirs(save_dir_bfm, exist_ok=True)
        # os.makedirs(save_dir_bfm_vis, exist_ok=True)
        # os.makedirs(save_dir_ldmk2d, exist_ok=True)
        # os.makedirs(save_dir_ldmk3d, exist_ok=True)
        
        print("Processing images from videos {}...".format(input_dir))
        align_dir = join(save_dir, "align_images")
        os.makedirs(align_dir, exist_ok=True)
        
        if len(os.listdir(align_dir)) >= process_frames:
            print(f"Already processed {len(os.listdir(align_dir))} frames. | skipping file.")
            return None
        else:
            print(f"Already processed {len(os.listdir(align_dir))} frames.")
        
        # ===> Read video or image files
        if video:
            video_stream = cv2.VideoCapture(input_dir)
            image_list = []
            total_images = 0
            while 1:
                if total_images < process_frames:
                    still_reading, frame = video_stream.read()
                    if not still_reading:
                        video_stream.release()
                        break
                    image_list.append(frame)
                    total_images += 1
                else:
                    break
        else:
            image_list = list(os.listdir(input_dir))
            image_list = [i for i in image_list if i.endswith('.png') or i.endswith('.jpg')]
            image_list.sort()
            
            frame_num = len(image_list)
            print("total {} images in {}".format(frame_num, input_dir))

            if frame_num == 0:
                print("{} images in {}, pass".format(frame_num, input_dir))
                return

        self.fd_ldmk_detector.reset()

        # ===> start processing
        idx = 0
        with torch.no_grad():
            for img_idx, img_item in tqdm(enumerate(image_list), desc=f"Processing images {idx} / {process_frames}"):
                idx += 1
                if img_idx > process_frames:
                    return None
                
                if video:
                    img = img_item
                else:
                    img_path = os.path.join(input_dir, img_item)
                    img = cv2.imread(img_path)
                
                ih, iw, c = img.shape
                
                # ===> detect ldmks NOTE: this is first farme ldmks detection
                try:
                    ldmks, ldmks_3d, boxes = self.fd_ldmk_detector.inference(cv2.imread(os.path.join(input_dir, image_list[0])))
                except Exception as e:
                    self.fd_ldmk_detector.reset()
                    print(e)
                    print("1")
                    return
                if not video:
                    self.fd_ldmk_detector.reset()
                
                ldmks = ldmks[:1] # only select first detected head
                ldmks_3d = ldmks_3d[:1]
                
                for i in range(len(ldmks)):
                    
                    # extract bfm params via Deep3DRecon
                    # input_recon = {'imgs':Image.fromarray(img[:,:,::-1]),'lms':ldmks[i]}
                    # bfm_params, trans_params = self.net_recon.forward(input_recon)
                    # self.net_recon.compute_visuals()
                    # visual = self.net_recon.output_vis.squeeze(0)
                    
                    # reverse y axis for later image cropping process
                    ldmk = ldmks[i].copy()
                    ldmk[:, -1] = ih - 1 - ldmk[:, -1]

                    ldmk_3d = ldmks_3d[i].copy()
                    ldmk_3d[:, 1] = ih - 1 - ldmk_3d[:, 1]

                    ldmk_2d_5pt = ldmks[i, self.stand_index]
                    ldmk_2d_5pt[:, -1] = ih - 1 - ldmk_2d_5pt[:, -1]
                    
                    # cropping
                    try:
                        trans_params, im_crop, ldmk_2d_5pt_crop, ldmk, ldmk_3d = align_img_bfm(Image.fromarray(img[:,:,::-1]), ldmk_2d_5pt, ldmk, ldmk_3d, target_size=self.target_size, rescale_factor=self.rescale_factor, index=img_idx, use_smooth=use_crop_smooth)
                    except Exception as e:
                        self.fd_ldmk_detector.reset()
                        print(e)
                        print("2")
                        return
                    
                    # reverse back to original y direction
                    ldmk = np.concatenate([ldmk[:, 0:1], self.target_size-ldmk[:, 1:2]], 1)
                    ldmk_3d = np.concatenate([ldmk_3d[:, 0:1], self.target_size-ldmk_3d[:, 1:2]], 1)   
                    
                    if video:
                        np.save(join(save_dir_ldmk2d, f'{str(img_idx+1).zfill(5)}.npy'), ldmk)
                        im_crop.save(join(align_dir, f'{str(img_idx+1).zfill(5)}.png'), quality=95)
                    else:
                        # np.save(join(save_dir_ldmk2d, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), ldmk)
                        # np.save(join(save_dir_ldmk3d, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), ldmk_3d)
                        # np.save(join(save_dir_bfm, img_item.replace(".png", ".npy").replace(".jpg", ".npy")), bfm_params)
                        # im_crop.save(img_path, quality=95)
                        im_crop.save(join(align_dir, f'{img_item}'), quality=95)
                        # Image.fromarray(visual.astype(np.uint8)).save(join(save_dir_bfm_vis, img_item), quality=95)
        
        test_end = time.time()
        
        print("process time is: {}".format(test_end - test_start))
        # self.fd_ldmk_detector.reset()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/audio2lip_process.yaml')
    parser.add_argument("--input_dir", type=str, 
                        default='/data/test-db/home/liyongyuan/Avatar_Teeth_enhance/sample/test/zhengkai1_demo/source')
    parser.add_argument("--save_dir", type=str, 
                        default='/data/test-db/home/liyongyuan/Avatar_Teeth_enhance/sample/test/zhengkai1_demo')
    parser.add_argument("--isvideo", type=bool, default=False)
    parser.add_argument("--ngpu", type=int, default=1)
    args = parser.parse_args()

    # init preprocess pipeline
    preprocessor = PreprocessImg(args)
    preprocessor._crop_extr_img(args.input_dir, args.save_dir, video=args.isvideo)
    
    
    # get file list
    # filelist = glob(args.input_dir + '/*.mp4')
    
    # for file in tqdm(filelist, desc='Processing'):
    #     preprocessor._crop_extr_img(file, args.save_dir, video=args.isvideo)
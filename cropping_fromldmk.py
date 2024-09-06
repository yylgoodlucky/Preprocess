
import torch, argparse, os, cv2, traceback
import numpy as np
import torch.nn.functional as F

from os.path import join, basename, dirname
from PIL import Image
from tqdm import tqdm

from kornia.geometry import warp_affine
from skimage import transform as trans
from concurrent.futures import ThreadPoolExecutor, as_completed
from process_slpt_landmarks import SLPTWrapper_singleframe

SLPT = SLPTWrapper_singleframe(modelpath='process_slpt_landmarks/weights/WFLW_6_layer.pth')

def draw_landmark(landmark: np.ndarray, image: np.ndarray):
    # for key, (x, y) in enumerate((landmark + 0.5).astype(np.int32)):
    #     cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # cv2.putText(image, str(key), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    landmark = landmark.astype(int)
    face_pts = list(range(0, 32))
    
    cv2.polylines(image, [landmark[face_pts]], isClosed=False, color=(0, 255, 0), thickness=2)
    return image

def get_motion_feature(imgs, lmks, crop_size=224, crop_len=16, reverse_y=False):
    """
    Return:
        imgs_warp          --torch.tensor  (N, 3, 224, 224)
    Parameters:
        imgs               --torch.tensor  (N, 3, H, W)
    """
    trans_m = estimate_norm_torch_pdfgc(lmks, imgs.shape[0], reverse_y=reverse_y)
    imgs_warp = warp_affine(imgs, trans_m, dsize=(224, 224))
    imgs_warp = imgs_warp[:,:,:crop_size - crop_len*2, crop_len:crop_size - crop_len]
    imgs_warp = torch.clamp(F.interpolate(imgs_warp,size=[crop_size,crop_size],mode='bilinear'),-1,1)

    return imgs_warp

def extract_3p_flame(lm):
    p0 = lm[60:68].mean(0)
    p1 = lm[68:75].mean(0)
    p2 = lm[76:89].mean(0)
    lm3p = np.stack([p0,p1,p2],axis=0) #(3,2)
    return lm3p

def estimate_norm_pdfgc(lm_70p, H, reverse_y=True):
    # modified from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (70, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_3p_flame(lm_70p)
    if reverse_y:
        lm[:, -1] = H - 1 - lm[:, -1]
    tform = trans.SimilarityTransform()
    src = np.array([[87,  59], [137,  59], [112, 120]], dtype=np.float32) # in size of 224
    tform.estimate(lm, src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]

def estimate_norm_torch_pdfgc(lm_70p, H, reverse_y=True):
    lm_70p_ = lm_70p.detach().cpu().numpy()
    M = []
    for i in range(lm_70p_.shape[0]):
        M.append(estimate_norm_pdfgc(lm_70p_[i], H, reverse_y=reverse_y))
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_70p.device)
    return M

def draw_landmark(landmark, image):
    for key, (x, y) in enumerate((landmark + 0.5).astype(np.int32)):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(key), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image

def gen_crop_img(input_dir, isvideo=True):
    basename_img = basename(input_dir).split('.')[0]
    dirname_img = dirname(input_dir).replace('video', 'image')
    save_dir = join(dirname_img, basename_img)
    os.makedirs(save_dir, exist_ok=True)

    # read video
    # print(f'[info]: processing {input_dir}')
    if isvideo:
        video_stream = cv2.VideoCapture(input_dir)
	
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)
    
    if len(frames) == os.listdir(save_dir):
        print(f'[info]: {input_dir} has been processed')
        return None
    
    for index, img in enumerate(frames):
        ldmk_3d = SLPT.get_lmark(img)
        # stand_index = np.array([96, 97, 54, 76, 82])
        # image = draw_landmark(ldmk_3d.astype(int)[stand_index,:], img)

        assert ldmk_3d is not None
        try:
            img_app = np.asarray(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            img_app = torch.from_numpy((img_app.astype(np.float32)/127.5 - 1)).to(device)
            img_app = img_app.permute([2,0,1]).unsqueeze(0)

            lmks_app = torch.from_numpy(ldmk_3d).to(device).unsqueeze(0)
            imgs_warp = get_motion_feature(img_app, lmks_app)

            img = (imgs_warp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{save_dir}/{str(index).zfill(5)}.png')
        except Exception as e:
            print(e)
            continue

def mp_handler(file):
    try:
        gen_crop_img(file)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, 
                        default='/data/test-db/home/liyongyuan/Datasets/avspeech/video')
    parser.add_argument("--filelist", type=str, 
                        default='/data/test-db/home/liyongyuan/Datasets/filelists/avspeech.txt')
    parser.add_argument("--np", type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda')

    # get image list
    filelist = []
    with open(args.filelist, 'r') as f:
        for line in f:
            line = line.strip()
            filelist.append(join(args.input_dir, line + '.mp4'))

    print('total videos :', len(filelist))
    
    # process multiple files
    if args.np > 1:
        process_num = args.np
        print('process_num: ', process_num)
        p_frames = ThreadPoolExecutor(process_num)
        futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in filelist]
        _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
        print("complete task!")

    filelist = ['/data/test-db/home/liyongyuan/Portrait-4D/asserts/test_avatar/0531_2_stand_cut.mp4']
    # process single video
    if args.np == 1:
        for file in tqdm(filelist):
            gen_crop_img(file)
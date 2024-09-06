import os
import numpy as np
import os.path as osp

from process_slpt_landmarks.utils import crop_v2, transform_pixel_v2
from process_slpt_landmarks.SLPT import Sparse_alignment_network
from process_slpt_landmarks.Config import cfg
from process_slpt_landmarks.utils.get_transforms import get_warp_mat

import torch, cv2, math
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import natsort
from tqdm import tqdm
from insightface.app import FaceAnalysis
from pathlib import Path
from natsort import natsorted


def draw_landmark(landmark: np.ndarray, image: np.ndarray):
    for key, (x, y) in enumerate((landmark + 0.5).astype(np.int32)):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # cv2.putText(image, str(key), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image


def crop_img(img: np.ndarray, bbox: np.ndarray, transform: np.ndarray):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1), math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans

class ProcessSLPTWrapper():
    def __init__(self, modelpath):
        super().__init__()

        self.modelpath = modelpath

        # self.fg_img_dir = img_dir
        # self.landmarks_dir = osp.join(osp.dirname(img_dir),  "landmarks")
        # os.makedirs(self.landmarks_dir, exist_ok=True)

        self.model = self._load_model()
        self.model.eval()
        
        self.app = FaceAnalysis(
            allowed_modules=["detection"], providers=["CUDAExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def _load_model(self):
        dir_path = osp.dirname(osp.realpath(__file__))
        model = Sparse_alignment_network(
            cfg.WFLW.NUM_POINT,
            cfg.MODEL.OUT_DIM,
            cfg.MODEL.TRAINABLE,
            cfg.MODEL.INTER_LAYER,
            cfg.MODEL.DILATION,
            cfg.TRANSFORMER.NHEAD,
            cfg.TRANSFORMER.FEED_DIM,
            osp.join(dir_path, cfg.WFLW.INITIAL_PATH),
            cfg,
        )
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

        checkpoint = torch.load(self.modelpath)
        pretrained_dict = {
            k: v for k, v in checkpoint.items() if k in model.module.state_dict().keys()
        }
        model.module.load_state_dict(pretrained_dict)

        return model

    def get_landmark(self, imgdir, fnames):
        
        lmarkdir = imgdir.replace('video', 'landmark98')
        os.makedirs(lmarkdir, exist_ok=True)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
        
        if len(fnames) == len(os.listdir(lmarkdir)):
            return None

        for fname in tqdm(fnames, total=len(fnames), desc="SPLT Landmarks"):
            frame = fname
            faces = self.app.get(frame)
            
            if (faces is None) or (len(faces) == 0):
                continue

            faces = sorted(faces, key=lambda d: d['det_score'], reverse=True)
            faces = sorted(faces, key=lambda d: d['det_score'], reverse=True)
            bbox = faces[0].bbox
            if bbox is None:
                continue

            bbox[0] = int(bbox[0] + 0.5)
            bbox[2] = int(bbox[2] + 0.5)
            bbox[1] = int(bbox[1] + 0.5)
            bbox[3] = int(bbox[3] + 0.5)
            alignment_input, trans = crop_img(frame.copy(), bbox, normalize)
            # if int(fname[:-4]) > 5: break
            # print(bbox)
            # # rgb = frame[:, :, ::-1].copy()
            # # alignment_input = torch.tensor(rgb[np.newaxis, ...]).permute(0, 3, 1, 2).float() / 255.0
            # # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # # alignment_input = normalize(alignment_input)

            with torch.no_grad():
                outputs_initial = self.model(alignment_input.cuda())
                output = outputs_initial[2][0, -1, :, :].cpu().numpy()

            landmark = transform_pixel_v2(
                output * cfg.MODEL.IMG_SIZE, trans, inverse=True
            )

            lmarkpath = osp.join(imgdir, fname).replace('video', 'landmark98') + '.npy'
            np.save(lmarkpath, landmark)

            # print(cfg.MODEL.IMG_SIZE)
            # frame = draw_landmark(landmark, frame)

            # stem_name = Path(fname).stem
            # np.savetxt(
            #     osp.join(self.landmarks_dir, f"{stem_name}.slpt"), landmark, "%f"
            # )

            if self.debugpath == None:
                continue

            if out_vid is None:
                out_vid = cv2.VideoWriter(
                    self.debugpath,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    25.0,
                    (frame.shape[1], frame.shape[0]),
                )

            out_vid.write(frame)

        if out_vid is not None:
            out_vid.release()

        # return self.anomaly_detection()
        return None

    def anomaly_detection(self):
        """
        Detects frames with large landmarks motion
        """

        lms_paths = natsorted(
            [f for f in os.listdir(self.landmarks_dir) if f.endswith(".slpt")]
        )
        full_lmss = [np.loadtxt(osp.join(self.landmarks_dir, f)) for f in lms_paths]

        cur_len = int(
            np.mean(
                np.max(np.max(full_lmss, axis=1) - np.min(full_lmss, axis=1), axis=1)
            )
        )
        cur_len = int(1.6 * cur_len)  # in order to include the neck area
        dst_size = min(cur_len, 512)  # avoid out of memory error
        cur_cxy = np.mean(
            (np.max(full_lmss, axis=1) + np.min(full_lmss, axis=1)) * 0.5, axis=0
        )
        cur_cxy[1] += cur_len * 0.1

        warp_scale, warp_trans, warp_mat = get_warp_mat(
            cur_cxy, cur_len / 2.0, float(dst_size)
        )

        for lmss in full_lmss:
            lms = lmss * warp_scale
            lms[:, 0] += warp_trans[0]
            lms[:, 1] += warp_trans[1]

            # margin = 0.15 * float(dst_size)
            # margin = 0.3 * float(dst_size)
            margin = 0.5 * float(dst_size)
            if (
                np.min(lms[:, 0]) < -margin
                or np.max(lms[:, 0]) > dst_size + margin
                or np.min(lms[:, 1]) < -margin
                or np.max(lms[:, 1]) > dst_size + margin
            ):
                return False, "Anomaly detected: landmarks are out of the bbox", 1

        return True, "Anomaly not detected", 0


def get_filelist(root, filedir):
        filelist = []
        with open(filedir, 'r') as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(root + line)

        return filelist


def mp_handler(file):
	try:
		wrapper.get_landmark(file)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()

def worker(file):
    try:
        wrapper.get_landmark(file)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import traceback
    import multiprocessing

    wrapper = ProcessSLPTWrapper(modelpath='/data/test-db/home/liyongyuan/SLPT-master/Weight/WFLW_6_layer.pth',
                                 debugpath=None)
    file = '/data/test-db/home/liyongyuan/Datasets/portrait4d_v2/group400/Clip+hGTmo09oqf8+P1+C0+F29502-29669/align_images'
    wrapper.get_landmark(file)
    
    # test_path = '/data/test-db/home/liyongyuan/Datasets/portrait4d_v2'
    # filelist_path = '/data/test-db/home/liyongyuan/Datasets/filelists/portrait4d_v2.txt'
    # filelists = get_filelist(test_path, filelist_path)
    
    
    # for file in tqdm(filelists):
    #     wrapper.get_landmark(file)
    
    
    # 多线程
    # process_num = 4
    # print('total file :', len(filelists))
    # print('process_num: ', process_num)
    
    # p_frames = ThreadPoolExecutor(process_num)
    
    # futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in filelists]
    # _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
    # print("complete task!")
    
    # 多进程
    # pool = multiprocessing.Pool(4)
    # pool.map(worker, filelists)

    # pool.close()
    # pool.join()
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
from torchvision.utils import make_grid
from torchvision.transforms.functional import normalize

app = FaceAnalysis(allowed_modules=["detection"], providers=["CUDAExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

normalize_s = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
normalize_s = transforms.Compose(
    [transforms.ToTensor(),
     normalize_s,]
)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


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

class SLPTWrapper_singleframe():
    def __init__(self, modelpath):
        super().__init__()

        self.modelpath = modelpath
        
        self.device_id = torch.cuda.current_device()

        self.model = self._load_model()
        self.model.eval()

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

        checkpoint = torch.load(self.modelpath, map_location=lambda storage, loc: storage.cuda(self.device_id))
        pretrained_dict = {
            k: v for k, v in checkpoint.items() if k in model.module.state_dict().keys()
        }
        model.module.load_state_dict(pretrained_dict)

        return model

    def get_lmark(self, image):
        faces = app.get(image)
        
        if (faces is None) or (len(faces) == 0):
            return None

        faces = sorted(faces, key=lambda d: d['det_score'], reverse=True)
        bbox = faces[0].bbox
        if bbox is None:
            return None

        bbox[0] = int(bbox[0] + 0.5)
        bbox[2] = int(bbox[2] + 0.5)
        bbox[1] = int(bbox[1] + 0.5)
        bbox[3] = int(bbox[3] + 0.5)
        alignment_input, trans = crop_img(image.copy(), bbox, normalize_s)
        # if int(fname[:-4]) > 5: break
        # print(bbox)
        # # rgb = frame[:, :, ::-1].copy()
        # # alignment_input = torch.tensor(rgb[np.newaxis, ...]).permute(0, 3, 1, 2).float() / 255.0
        # # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # # alignment_input = normalize(alignment_input)

        with torch.no_grad():
            outputs_initial = self.model(alignment_input.cuda())
            output = outputs_initial[2][0, -1, :, :].cpu().numpy()

        landmark = transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
        
        return landmark


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

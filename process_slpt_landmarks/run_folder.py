import numpy as np
import argparse
import os.path

from Config import cfg
from Config import update_config

from NeRF.preprocess.process_slpt_landmarks.SLPT import Sparse_alignment_network

import torch, cv2, math

from pathlib import Path

import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import utils
import natsort
from tqdm import tqdm
from insightface.app import FaceAnalysis
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Video Demo')

    # face detector
    parser.add_argument('-m', '--trained_model', default='./Weight/Face_Detector/yunet_final.pth',
                        type=str, help='Trained state_dict file path to open')
    
    parser.add_argument('--confidence_threshold', default=0.7, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.3, type=float, help='visualization_threshold')
    parser.add_argument('--base_layers', default=16, type=int, help='the number of the output of the first layer')
    parser.add_argument('--device', default='cuda:0', help='which device the program will run on. cuda:0, cuda:1, ...')
    # landmark detector
    parser.add_argument('--modelDir', help='model directory', type=str, default='./Weight')
    parser.add_argument('--checkpoint', help='checkpoint file', type=str, default='models/models_slpt/WFLW_6_layer.pth')
    parser.add_argument('--logDir', help='log directory', type=str, default='./log')
    parser.add_argument('--dataDir', help='data directory', type=str, default='./')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default=None)
    parser.add_argument('--img_step', default=1, type=int, help='the image file to be detected')
    parser.add_argument('--input_folder', default='./input', type=str, help='the image file to be detected')
    parser.add_argument('--output_folder', default='./output', type=str, help='the image file to be detected')
    parser.add_argument('--debug_path', default='none', type=str, help='nms_threshold')
    args = parser.parse_args()

    return args


def draw_landmark(landmark, image):

    for (x, y) in (landmark + 0.5).astype(np.int32):
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    return image

def crop_img(img, bbox, transform):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2
    center = np.array([cx, cy])

    scale = max(math.ceil(x2) - math.floor(x1),
                math.ceil(y2) - math.floor(y1)) / 200.0

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    input, trans = utils.crop_v2(img, center, scale * 1.15, (256, 256))

    input = transform(input).unsqueeze(0)

    return input, trans

if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)

    device = torch.device(args.device)

    torch.set_grad_enabled(False)

    # Cuda
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model = Sparse_alignment_network(cfg.WFLW.NUM_POINT, cfg.MODEL.OUT_DIM,
                                     cfg.MODEL.TRAINABLE, cfg.MODEL.INTER_LAYER,
                                     cfg.MODEL.DILATION, cfg.TRANSFORMER.NHEAD,
                                     cfg.TRANSFORMER.FEED_DIM, os.path.join(dir_path, cfg.WFLW.INITIAL_PATH), cfg)
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    checkpoint_file = os.path.join(dir_path, args.checkpoint)
    checkpoint = torch.load(checkpoint_file)
    pretrained_dict = {k: v for k, v in checkpoint.items()
                       if k in model.module.state_dict().keys()}
    model.module.load_state_dict(pretrained_dict)
    model.eval()

    print('Finished loading face landmark detector')


    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    normalize = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    app = FaceAnalysis(allowed_modules=['detection'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(224, 224))

    src_dir = args.input_folder
    dst_dir = args.output_folder
    debug_path = args.debug_path
    out_vid = None
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    image_paths = natsort.natsorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
    for img_path in tqdm(image_paths, desc='detect landmarks'):
        frame = cv2.imread(os.path.join(src_dir, img_path))
        faces = app.get(frame, max_num=1)
        if (faces is None) or (len(faces) == 0):
            continue
        bbox = faces[0].bbox
        if bbox is None:
            continue
        bbox[0] = int(bbox[0]  + 0.5)
        bbox[2] = int(bbox[2]  + 0.5)
        bbox[1] = int(bbox[1]  + 0.5)
        bbox[3] = int(bbox[3]  + 0.5)
        alignment_input, trans = crop_img(frame.copy(), bbox, normalize)
        outputs_initial = model(alignment_input.cuda())
        output = outputs_initial[2][0, -1, :, :].cpu().numpy()
        landmark = utils.transform_pixel_v2(output * cfg.MODEL.IMG_SIZE, trans, inverse=True)
        frame = draw_landmark(landmark, frame)
        np.savetxt(os.path.join(dst_dir, img_path[:-3] + 'slpt_new'), landmark, '%f')
        if debug_path == 'none':
            continue
        if out_vid is None:
            out_vid = cv2.VideoWriter(os.path.join(debug_path, 'wflw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (frame.shape[1], frame.shape[0]))
        out_vid.write(frame)
    if out_vid is not None:
        out_vid.release()


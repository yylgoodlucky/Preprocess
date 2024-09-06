import os, argparse, cv2
import torch
import torch.nn as nn 
from audio2motion import FanEncoder

class Extractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.net_motion = FanEncoder(self.args.pose_dim, self.args.eye_dim)

    def load_model(self):
        model_dict = torch.load(self.args.motion_checkpoint)
        init_dict = self.net_motion.state_dict()
        key_list = list(set(model_dict.keys()).intersection(set(init_dict.keys())))
        for k in key_list:
            
            if "mouth_fc" in k or "headpose_fc" in k or "classifier" in k or "to_feature" in k or "to_embed" in k:
                continue
                
            init_dict[k] = model_dict[k]
        self.net_motion.load_state_dict(init_dict)

    def _extract_motion_dict(self, image_list):
        motion = {}
        self.load_model()
        headpose_emb, eye_embed, emo_embed, mouth_feat = self.net_motion(image_list)

        motion['headpose_emb'] = headpose_emb
        motion['eye_embed'] = eye_embed
        motion['emo_embed'] = emo_embed
        motion['mouth_feat'] = mouth_feat

        return motion

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_checkpoint', type=str, 
                        default='/dellnas/nas-1/users/liyongyuan/workspace/audio2expression/checkpoints/motion_model.pth')
    parser.add_argument('--pose_dim', type=int, default=6)
    parser.add_argument('--eye_dim', type=int, default=6)
    
    args = parser.parse_args()

    img = '/dellnas/nas-1/users/liyongyuan/workspace/PD-FGC-inference/data/motions/exp/vox2_id01000_00077/000001.jpg'
    # img_tensor = torch.from_numpy(cv2.imread(img)/255., dtype=torch.float64).squeeze(0)
    img_tensor = torch.randn(24, 3, 224, 224)

    extr_motion = Extractor(args)
    motion_dict = extr_motion._extract_motion_dict(img_tensor)  # forward time 1.96
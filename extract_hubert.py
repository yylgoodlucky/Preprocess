# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
# fmt: off
import numpy as np
import torch, os
# fmt: on
import librosa
import fairseq
import soundfile as sf
import torch.nn.functional as F
import multiprocessing, traceback

from os.path import join, dirname, basename, splitext
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


class HubertFeatureReader:
    """
    Wrapper class to run inference on HuBERT model.
    Helps extract features for a given audio file.
    """

    def __init__(self, checkpoint_path, layer=-1, max_chunk=1600000):
        (
            model,
            cfg,
            task,
        ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path]
        )
        self.model = model[0].eval().cuda()
        
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.pad = torch.zeros((1, 80)).cuda()

    def read_audio(self, path: str, ref_len=None):
        wav, sr = librosa.load(path, sr=16000)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        assert sr == self.task.cfg.sample_rate, sr
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            print(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, file_path, ref_len=None):
        x = self.read_audio(file_path, ref_len)
        
        ## NOTE scaled
        # scale = 1.5
        scale = 1.0
        x *= scale

        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                x_chunk = torch.cat([x_chunk, self.pad], dim=1)
                
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk.detach().clone().cpu())
        
        feat_torch = torch.cat(feat, 1).squeeze(0)
        
        feats = feat_torch.numpy()
        if feats.shape[0] % 2 == 1:
            feats = feats[:-1, :]
        # feats = np.reshape(feats, (feats.shape[0]//2, 2 * feats.shape[1]))

        return feats

def get_filelists(data_root, file_path):
    filelist = []
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(join(data_root, line))
        
    return filelist


def _extract_hubert(vfile, reader):
    """ Extract features from audio file using HuBERT model. 
    """
    audio_path = vfile.replace('dev', 'audio').replace('.mp4', '.wav')
    save_path = vfile.replace('dev', 'hubert').replace('.mp4', '.npy')
    if os.path.exists(save_path): return None
    if not os.path.exists(dirname(save_path)): os.makedirs(dirname(save_path), exist_ok=True)

    feats = reader.get_feats(audio_path)
    
    np.save(save_path, feats)


def mp_handler_hubert(job):
    """ Handle audio file processing using HuBERT model.
    """
    vfile, net = job
    try:
        _extract_hubert(vfile, net)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()
        
def main(args, net):
    
    print("looking up paths.... from ", args.data_root)
    filelists = get_filelists(args.data_root, args.filelist)
    
    jobs = [(vfile, net) for vfile in filelists]
    p_audio = ProcessPoolExecutor(args.process_num)
    futures_audio = [p_audio.submit(mp_handler_hubert, j) for j in jobs]

    _ = [r.result() for r in tqdm(as_completed(futures_audio), total=len(futures_audio))]
    print("[info] : feature extraction completed.")

def single_precess(args, net):
    """ Perform a single precess operation on a given.
    """
    print("looking up paths.... from ", args.data_root)
    filelists = get_filelists(args.data_root, args.filelist)
    
    for file in tqdm(filelists):
        _extract_hubert(file, net)
        
    print("[info] : feature extraction completed.")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/data/test-db/home/liyongyuan/Datasets/VoxCeleb2',
                        help='path to audio file')
    parser.add_argument('--filelist', type=str, default='/data/test-db/home/liyongyuan/Datasets/filelists/VoxCeleb2.txt')
    parser.add_argument('--model', type=str, default='/data/test-db/home/liyongyuan/A_Research/adnerf_model/models/hubert-large/chinese-hubert-large.pt')
    parser.add_argument('--process_num', type=int, default=1, help='Number of processes to use')
    
    opt = parser.parse_args()
    reader = HubertFeatureReader(checkpoint_path=opt.model, layer=-1, max_chunk=1600000)

    # mutliple process
    if opt.process_num > 1:
        main(opt, reader)
    
    # single process
    if opt.process_num == 1:
        single_precess(opt, reader)
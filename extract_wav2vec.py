import os
import librosa
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

from os.path import join, dirname, basename
from transformers import Wav2Vec2Processor
from lib.wav2vec import Wav2Vec2Model
from moviepy.editor import VideoFileClip


# ===> Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english')
# ===> audio feature extraction
audio_encoder = Wav2Vec2Model.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-english').to(device='cuda')
# wav2vec 2.0 weights initialization
audio_encoder.feature_extractor._freeze_parameters()
        
if __name__ == '__main__':
    dpath = '/data/test-db/home/liyongyuan/Datasets/height_crop_bilibili/clip_videos'
    
    save_path = '/data/test-db/home/liyongyuan/Datasets/height_crop_bilibili/preprocessed'
    wav2vec_path = join(save_path, 'wav2vec')
    os.makedirs(wav2vec_path, exist_ok=True)
    
    videos_list = glob(join(dpath, '*.mp4'))
    
    for video_path in tqdm(videos_list, desc="extact wav2vec"):

        audio = VideoFileClip(video_path).audio
        audio.write_audiofile('./audio.wav')

        # init save path
        test_name = os.path.basename(video_path).split(".")[0]
        wav2vec_feat_path = os.path.join(wav2vec_path, test_name+'.npy')
        
        speech_array, _ = librosa.load('./audio.wav', sr=16000)
        audio_feature = np.squeeze(processor(speech_array, sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0])).astype(np.float32)
        audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')

        hidden_states = audio_encoder(audio_feature, 'multi').last_hidden_state

        np.save(wav2vec_feat_path, hidden_states)
        
        
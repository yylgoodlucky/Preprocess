
import os
from glob import glob
from tqdm import tqdm
from os.path import join, basename, dirname

command = '/usr/bin/ffmpeg -i {} -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -loglevel quiet -segment_time 00:00:10 -f segment {}_%03d.mp4'

if __name__ == '__main__':
    dpath = '/data/test-db/home/liyongyuan/Datasets/height_crop_tiktok/videos'
    videos_list = glob(join(dpath, '*.mp4'))
    
    for video_path in tqdm(videos_list):
        os.system(command.format(video_path, video_path[:-4].replace('videos', 'clip_videos')))
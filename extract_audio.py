import os, argparse
import subprocess, traceback

from tqdm import tqdm
from os.path import join, basename, dirname, splitext
from concurrent.futures import ProcessPoolExecutor, as_completed


commend = 'ffmpeg -loglevel quiet -y -i {} -f wav -ar 16000 {}'


def get_filelists(data_root, file_path):
    filelist = []
    with open(file_path, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            filelist.append(join(data_root, line))
        
    return filelist

def process_audio_file(vfile):
    """ Extract audio from video file. """
    save_audio_path = vfile.replace('dev', 'audio').replace('.mp4', '.wav')
    save_dir = dirname(save_audio_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
    
    subprocess.call(commend.format(vfile, save_audio_path), shell=True)


def mp_handler_audio(vfile):
    try:
        process_audio_file(vfile)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()

def main(args):
    print("looking up paths.... from ", args.data_root)
    filelists = get_filelists(args.data_root, args.filelist)
    

    p_audio = ProcessPoolExecutor(args.process_num)
    futures_audio = [p_audio.submit(mp_handler_audio, vfile) for vfile in filelists]

    _ = [r.result() for r in tqdm(as_completed(futures_audio), total=len(futures_audio))]
    print("[info] : audio extraction completed.")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract audio from video')
    parser.add_argument('--data_root', type=str, 
                        default='/data/test-db/home/liyongyuan/Datasets/VoxCeleb2', help='Input directory containing videos')
    parser.add_argument('--filelist', type=str, 
                        default='/data/test-db/home/liyongyuan/Datasets/filelists/VoxCeleb2.txt', help='Filelist containing video names')
    parser.add_argument('--process_num', type=int, default=4, help='Number of processes to use')
    args = parser.parse_args()
    
    main(args)
    
    
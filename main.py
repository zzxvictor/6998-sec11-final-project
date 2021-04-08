import argparse
import pathlib
from system.video_streaming import  VideoReader

# python main.py -vp I:\Data\CNR-EXT-Patches-150x150\video.mp4 -ap I:\Data\CNR-EXT-Patches-150x150\video_annotation.txt
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vp', dest='vp', type=pathlib.Path)
    parser.add_argument('-ap', dest='ap', type=pathlib.Path)
    args = parser.parse_args()
    reader = VideoReader(args.vp, args.ap)
    reader.start()
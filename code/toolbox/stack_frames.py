import os
import cv2
import argparse
from tqdm import tqdm

from utils import IMAGE_EXT

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stacking the frames into an video.")
    parser.add_argument("frame_dir", type=str, help="path of the frames")
    parser.add_argument("video_pth", type=str, help="path for saving the output video")
    parser.add_argument("--fps", type=int, default=25, help="frames per seconds (FPS)")
    parser.add_argument("--height", type=int, default=None, help="height of the output video")
    parser.add_argument("--width", type=int, default=None, help="width of the output video")

    args = parser.parse_args()

    frames = filter(lambda x: os.path.splitext(x)[-1] in IMAGE_EXT, sorted(os.listdir(args.frame_dir)))
    frame_pths = [os.path.join(args.frame_dir, p) for p in frames]

    H, W = (args.height, args.width) if args.height and args.width else cv2.imread(frame_pths[0]).shape[: 2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.video_pth, fourcc, args.fps, (W, H))

    try:
        for frame_pth in tqdm(frame_pths):
            video.write(cv2.resize(cv2.imread(frame_pth), (W, H)))
    except Exception as exception:
        raise exception
    finally:
        video.release()

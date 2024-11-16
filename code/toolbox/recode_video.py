import argparse
from moviepy.editor import VideoFileClip

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Re-coding the input video.")
    parser.add_argument("input", type=str, help="path of the input video")
    parser.add_argument("output", type=str, help="path of the output video")
    parser.add_argument("--codec", type=str, default='libx264', help="video codec")

    args = parser.parse_args()

    clip = VideoFileClip(args.input)

    try:
        clip.write_videofile(args.output, codec=args.codec)
    except Exception as exception:
        raise exception
    finally:
        clip.close()

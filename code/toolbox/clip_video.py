import argparse
from moviepy.editor import VideoFileClip

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Clipping the input video.")
    parser.add_argument("input", type=str, help="path of the input video")
    parser.add_argument("output", type=str, help="path of the output video")
    parser.add_argument("--start", type=int, default=0, help="start time (in second)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--end", type=int, default=None, help="end time (in second)")
    group.add_argument("--duration", type=int, default=None, help="duration (in second)")

    args = parser.parse_args()

    clip = VideoFileClip(args.input)

    try:
        if args.duration:
            cut_clip = clip.subclip(args.start, args.duration)
        else:
            cut_clip = clip.subclip(args.start, t_end=args.end)
        cut_clip.write_videofile(args.output)
    except Exception as exception:
        raise exception
    finally:
        clip.close()

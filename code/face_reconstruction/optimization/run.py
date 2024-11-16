import os
import argparse
import subprocess
from tqdm import tqdm
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool

from utils import FACE_DATASETS, FACE_DATA_ROOT
from utils import FACE_META_ROOT, FACE_META_FILE
from utils import Timer

command = 'python -m faceverse.tracking_offline --device {} --video {} --output {} --save_coeff --no_pbar'

def parsing(datasets):
    vids = []
    for dataset in datasets:
        with open(os.path.join(FACE_META_ROOT, FACE_META_FILE[dataset]), 'r') as f:
            vids.extend(f.readlines())

    vids = tqdm(vids, dynamic_ncols=True, desc='loading videos')
    return [os.path.join(FACE_DATA_ROOT, vid.strip()) for vid in vids]

def exec(command, device, video):
    try:
        command = command.format(device, video, video)
        subprocess.run(command, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, None
    except subprocess.CalledProcessError as exception:
        return False, str(exception) + f'video: {video}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3DMM coeffs extraction.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--num_workers', type=int, default=4, help='num of threads in parallel')

    args = parser.parse_args()

    timestamp = str(datetime.now())[:-7].replace(' ', '-')

    videos = parsing(FACE_DATASETS if 'all' in args.datasets else args.datasets)

    timer = Timer()
    timer.start()

    with ThreadPool(processes=args.num_workers) as pool:
        results = pool.map(partial(exec, command, args.device), videos)

    for success, info in results:
        if not success:
            with open(f'{timestamp}.error', 'a') as f:
                f.write(info + '.\n')

    h, m, s = timer.consume()
    print(f'Time consuming: {h}h {m}m {s}s')

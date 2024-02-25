import argparse
import cv2
import numpy as np
import subprocess
import os
import requests
import sys
import torch
from deoldify.visualize import get_image_colorizer
from PIL import Image

def calculate_mean_std(x):
    return np.mean(x), np.std(x)

def download_file(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = f"ffmpeg -i \"{video_path}\" \"{output_dir}/frame_%04d.png\""
    subprocess.run(cmd, shell=True, check=True)

def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    l_source, a_source, b_source = cv2.split(source)
    l_target, a_target, b_target = cv2.split(target)

    l_mean_source, l_std_source = calculate_mean_std(l_source)
    a_mean_source, a_std_source = calculate_mean_std(a_source)
    b_mean_source, b_std_source = calculate_mean_std(b_source)

    l_mean_target, l_std_target = calculate_mean_std(l_target)
    a_mean_target, a_std_target = calculate_mean_std(a_target)
    b_mean_target, b_std_target = calculate_mean_std(b_target)

    l_source -= l_mean_source
    a_source -= a_mean_source
    b_source -= b_mean_source

    l_source *= (l_std_target / l_std_source)
    a_source *= (a_std_target / a_std_source)
    b_source *= (b_std_target / b_std_source)

    l_source += l_mean_target
    a_source += a_mean_target
    b_source += b_mean_target

    transfer = cv2.merge([l_source, a_source, b_source])
    transfer = cv2.cvtColor(np.clip(transfer, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer

def apply_color_transfer(previous_frame, frame, l_mean_first, l_std_first):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype("float32")
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2LAB).astype("float32")

    l, a, b = cv2.split(frame)
    l_prev, a_prev, b_prev = cv2.split(previous_frame)

    l_mean, l_std = calculate_mean_std(l)
    l_mean_prev, l_std_prev = calculate_mean_std(l_prev)

    l -= l_mean
    l *= (l_std_first / l_std)
    l += l_mean_first

    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(np.clip(transfer, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

    return transfer

def colorize_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    previous_frame = None

    colorizer = get_image_colorizer(artistic=True)

    first_frame = cv2.imread(frames[0])
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2LAB).astype("float32")
    l_mean_first, l_std_first = calculate_mean_std(cv2.split(first_frame)[0])

    for frame in frames:
        result = colorizer.get_transformed_image(frame, render_factor=35, watermarked=False) 

        if previous_frame is not None:
            result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            previous_frame = cv2.cvtColor(np.array(previous_frame), cv2.COLOR_RGB2BGR)
            result = apply_color_transfer(previous_frame, result, l_mean_first, l_std_first)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(frame)), cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))
        previous_frame = Image.fromarray(cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB))

def reassemble_video(input_dir, output_video):
    cmd = f"ffmpeg -r 24 -i \"{input_dir}/frame_%04d.png\" -vcodec libx264 \"{output_video}\""
    subprocess.run(cmd, shell=True, check=True)

def download_model():
    url = "https://huggingface.co/spaces/aryadytm/photo-colorization/resolve/main/models/ColorizeArtistic_gen.pth?download=true"
    filename = "models/ColorizeArtistic_gen.pth"
    download_file(url, filename)

def main(args):
    os.makedirs('models', exist_ok=True)
    video_path = args.file
    raw_frames_dir = "raw_frames"
    colorized_frames_dir = "colorized_frames"
    output_video = "colorized_video.mp4"
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Assuming only one GPU, adjust if necessary
        print("Using GPU")
    else:
        print("Using CPU")

    download_model()
    extract_frames(video_path, raw_frames_dir)
    colorize_frames(raw_frames_dir, colorized_frames_dir)
    reassemble_video(colorized_frames_dir, output_video)

def parse_args():
    parser = argparse.ArgumentParser(description="Colorize video frames")
    parser.add_argument("--file", type=str, required=True, help="Path to the video file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

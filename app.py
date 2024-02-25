import argparse
import cv2
import numpy as np
import subprocess
import os
import requests
import sys
import torch
from deoldify import device 
from deoldify.visualize import get_image_colorizer
from PIL import Image

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

    # Compute means and standard deviations for each channel
    lMeanSrc, aMeanSrc, bMeanSrc = np.mean(source, axis=(0, 1))
    lMeanTar, aMeanTar, bMeanTar = np.mean(target, axis=(0, 1))

    lStdSrc, aStdSrc, bStdSrc = np.std(source, axis=(0, 1))
    lStdTar, aStdTar, bStdTar = np.std(target, axis=(0, 1))

    # Subtract the means from the source image
    source[:,:,0] -= lMeanSrc
    source[:,:,1] -= aMeanSrc
    source[:,:,2] -= bMeanSrc

    # Scale by the standard deviations
    source[:,:,0] *= (lStdTar / lStdSrc)
    source[:,:,1] *= (aStdTar / aStdSrc)
    source[:,:,2] *= (bStdTar / bStdSrc)

    # Add the target means
    source[:,:,0] += lMeanTar
    source[:,:,1] += aMeanTar
    source[:,:,2] += bMeanTar

    # Clip the values to the [0, 255] range, convert back to uint8
    return cv2.cvtColor(np.clip(source, 0, 255).astype("uint8"), cv2.COLOR_LAB2BGR)

def colorize_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    previous_frame = None

    colorizer = get_image_colorizer(artistic=True, watermarked=False)

    for frame in frames:
        result = colorizer.get_transformed_image(frame, render_factor=35)  # Adjust render_factor if needed

        if previous_frame is not None:
            result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            previous_frame = cv2.cvtColor(np.array(previous_frame), cv2.COLOR_RGB2BGR)
            result = color_transfer(previous_frame, result)
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

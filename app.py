import argparse
import cv2
import numpy as np
import subprocess
import os
import requests
import sys
import torch
from deoldify import device
from deoldify.visualize import VideoColorizer, ModelImageVisualizer, get_image_colorizer
from transformers import DeiTFeatureExtractor, DeiTForImageClassification
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
    cmd = f"ffmpeg -i {video_path} {output_dir}/frame_%04d.png"
    subprocess.run(cmd, shell=True)


def color_transfer(source, target):
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

    lMeanSrc = np.mean(source[:,:,0])
    lMeanTar = np.mean(target[:,:,0])

    lStdSrc = np.std(source[:,:,0])
    lStdTar = np.std(target[:,:,0])

    source[:,:,0] = ((source[:,:,0]-lMeanSrc)*(lStdTar/lStdSrc)) + lMeanTar

    return cv2.cvtColor(source.astype("uint8"), cv2.COLOR_LAB2BGR)

def colorize_frames(input_dir, output_dir):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    previous_frame = None

    # Create an ImageColorizer object
    colorizer = get_image_colorizer(artistic=True)

    for frame in frames:
        # Colorize image with DeOldify
        result = colorizer.get_transformed_image(frame)

        # If previous_frame is not None, adjust colors to match previous frame
        if previous_frame is not None:
            result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            previous_frame = cv2.cvtColor(np.array(previous_frame), cv2.COLOR_RGB2BGR)
            result = color_transfer(previous_frame, result)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        # Convert numpy array to PIL Image
        result_image = Image.fromarray(result)

        # Save colorized frame
        result_image.save(os.path.join(output_dir, os.path.basename(frame)))
        previous_frame = result

def reassemble_video(input_dir, output_video):
    cmd = f"ffmpeg -r 24 -i {input_dir}/frame_%04d.png -vcodec libx264 {output_video}"
    subprocess.run(cmd, shell=True)


def download_model():
    # URL of the pre-trained model
    url = "https://huggingface.co/spaces/aryadytm/photo-colorization/resolve/main/models/ColorizeArtistic_gen.pth?download=true"

    # Path where the pre-trained model will be saved
    filename = "models/ColorizeArtistic_gen.pth"

    download_file(url, filename)

def main(args):
    os.makedirs('models', exist_ok=True)
    video_path = args.file
    raw_frames_dir = "raw_frames"
    colorized_frames_dir = "colorized_frames"
    output_video = "colorized_video.mp4"
    
    render_factor = 35  # You may need to adjust this value
    model = get_image_colorizer(artistic=True)
    vis = ModelImageVisualizer(model, results_dir='./')

    download_model()  # Download the model
    extract_frames(video_path, raw_frames_dir)
    colorize_frames(raw_frames_dir, colorized_frames_dir)  # Remove colorizer from here
    reassemble_video(colorized_frames_dir, output_video)


def parse_args():
    parser = argparse.ArgumentParser(description="Colorize video frames")
    parser.add_argument("--file", type=str, required=True, help="Path to the video file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

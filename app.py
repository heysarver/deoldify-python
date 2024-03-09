import argparse
import cv2
import numpy as np
import subprocess
import os
import requests
import sys
import time
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image
from deoldify import device
from deoldify.device_id import DeviceId
device.set(device=DeviceId.GPU0) # this has to be set before the next from/import
from deoldify.visualize import get_image_colorizer, get_video_colorizer


def check_cuda():
    if torch.cuda.is_available():
        print("CUDA device is available.")
        return True
    else:
        print("No CUDA device available.")
        return False
    

def calculate_mean_std(x):
    return np.mean(x), np.std(x)


def download_file(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))

        progress_bar = tqdm(total=total, unit='iB', unit_scale=True)
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            progress_bar.update(size)
        progress_bar.close()

        if total != 0 and progress_bar.n != total:
            print("ERROR, something went wrong")


def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmd = f"ffmpeg -hwaccel nvdec -i \"{video_path}\" \"{output_dir}/frame_%09d.png\""
    subprocess.run(cmd, shell=True, check=True)


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


def colorize_frames(input_dir, output_dir, independent_colorization=False):
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')])
    previous_frame = None

    colorizer = get_image_colorizer(artistic=False, render_factor=35)

    first_frame = cv2.imread(frames[0])
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2LAB).astype("float32")
    l_mean_first, l_std_first = calculate_mean_std(cv2.split(first_frame)[0])

    for frame in tqdm(frames, desc="Colorizing frames", unit="frame"):
        result = colorizer.get_transformed_image(frame, render_factor=35, watermarked=False)

        if previous_frame is not None and not independent_colorization:
            result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            previous_frame_bgr = cv2.cvtColor(np.array(previous_frame), cv2.COLOR_RGB2BGR)
            result = apply_color_transfer(previous_frame_bgr, result_bgr, l_mean_first, l_std_first)
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(output_dir, os.path.basename(frame)), cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR))
        previous_frame = Image.fromarray(cv2.cvtColor(np.array(result), cv2.COLOR_BGR2RGB))


def reassemble_video(input_dir, output_video, original_video, output_dir, force_overwrite_output):
    os.makedirs(output_dir, exist_ok=True)
    output_video_path = os.path.join(output_dir, output_video)
    
    if os.path.exists(output_video_path) and force_overwrite_output:
        os.remove(output_video_path)
    
    temp_video = os.path.join(output_dir, "temp_video.mp4")
    cmd_video = f"ffmpeg -r 24 -i \"{input_dir}/frame_%09d.png\" -vcodec h264_nvenc \"{temp_video}\""
    subprocess.run(cmd_video, shell=True, check=True)

    cmd_audio = f"ffmpeg -hwaccel nvdec -i \"{original_video}\" -vn -acodec aac original_audio.aac"
    subprocess.run(cmd_audio, shell=True, check=True)

    cmd_merge = f"ffmpeg -hwaccel nvdec -i \"{temp_video}\" -i original_audio.aac -c:v copy -c:a aac \"{output_video_path}\""
    subprocess.run(cmd_merge, shell=True, check=True)

    os.remove(temp_video)
    os.remove("original_audio.aac")


def download_model(model_type, force_download):
    model_urls = {
        "image": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeStable_gen.pth?download=true",
        "video": "https://huggingface.co/spensercai/DeOldify/resolve/main/ColorizeVideo_gen.pth?download=true"
    }
    model_files = {
        "image": "models/ColorizeStable_gen.pth",
        "video": "models/ColorizeVideo_gen.pth"
    }

    url = model_urls[model_type]
    filename = model_files[model_type]

    if os.path.exists(filename) and not force_download:
        print(f"Model file {filename} already exists. Skipping download.")
    else:
        download_file(url, filename)


def parse_args():
    parser = argparse.ArgumentParser(description="Colorize video frames")
    parser.add_argument("--file", type=str, default=os.getenv('FILE'), help="Path to the video file")
    parser.add_argument("--force-download-model", action='store_true', default=os.getenv('FORCE_DOWNLOAD_MODEL', False), help="Force download the model even if it exists")
    parser.add_argument("--force-overwrite-output", action='store_true', default=os.getenv('FORCE_OVERWRITE_OUTPUT', False), help="Force overwrite the output file even if it exists")
    parser.add_argument("--independent-colorization", action='store_true', default=os.getenv('INDEPENDENT_COLORIZATION', False), help="Colorize each frame independently without applying color transfer")
    parser.add_argument("--use-video-colorizer", action='store_true', default=os.getenv('USE_VIDEO_COLORIZER', False), help="Use video colorizer instead of image colorizer")
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    check_cuda()

    args = parse_args()
    
    os.makedirs('models', exist_ok=True)
    input_video = args.file
    force_download_model = args.force_download_model
    force_overwrite_output = args.force_overwrite_output
    independent_colorization = args.independent_colorization
    use_video_colorizer = args.use_video_colorizer
    raw_frames_dir = "raw_frames"
    colorized_frames_dir = "colorized_frames"
    output_video = "colorized_video.mp4"
    output_dir = "output"
    
    if use_video_colorizer:
        download_model("video", force_download=force_download_model)
        colorizer = get_video_colorizer(render_factor=35)
        colorizer.colorize_from_file_name(input_video, render_factor=35, watermarked=False)
    else:
        download_model("image", force_download=force_download_model)
        extract_frames(input_video, raw_frames_dir)
        colorize_frames(raw_frames_dir, colorized_frames_dir, independent_colorization)
        reassemble_video(colorized_frames_dir, output_video, input_video, output_dir, force_overwrite_output)

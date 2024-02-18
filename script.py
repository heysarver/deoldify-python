from deoldify.visualize import get_video_colorizer
import sys
import os
import subprocess
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
import torch
import fastai
from pathlib import Path

# Command line arguments
video_path = sys.argv[1]
render_factor = int(sys.argv[2]) if len(sys.argv) > 2 else 21
watermarked = sys.argv[3].lower() == 'true' if len(sys.argv) > 3 else True

# Setup
device.set(device=DeviceId.GPU0)

if not torch.cuda.is_available():
    print('GPU not available.')

# Check if models directory exists, if not create it
if not os.path.isdir('models'):
    os.makedirs('models')

# Download the model if it doesn't exist
model_url = 'https://data.deepai.org/deoldify/ColorizeVideo_gen.pth'
model_path = './models/ColorizeVideo_gen.pth'
if not os.path.isfile(model_path):
    subprocess.run(['wget', model_url, '-O', model_path])

colorizer = get_video_colorizer()

# Colorize
if os.path.isfile(video_path):
    colorized_video_path = colorizer.colorize_from_path(
        video_path, render_factor, watermarked=watermarked)
    print(f'Colorized video saved at {colorized_video_path}')
else:
    print('Invalid video path. Please provide a valid video file path.')

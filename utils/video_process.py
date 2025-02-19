import os
import cv2  
import time
import base64
import random
import numpy as np
import google.generativeai as genai
import hashlib
import requests
import numpy.typing as npt
import json

def download_video(video_url, video_tmp_dir = "/gpfs/radev/home/yz979/scratch/video_cache"):
    video_id = hashlib.md5(video_url.encode()).hexdigest()
    video_subdir = os.path.join(video_tmp_dir, video_id)
    os.makedirs(video_subdir, exist_ok=True)

    video_path = os.path.join(video_subdir, f"video.mp4")
    if not os.path.exists(video_path):
        with open(video_path, "wb") as f:
            response = requests.get(video_url)
            f.write(response.content)

    return video_path, video_id

def prepare_base64frames(model_name, video_url, total_frames, video_tmp_dir):
    video_path, video_id = download_video(video_url)

    image_subdir = os.path.join(video_tmp_dir, video_id, f"{total_frames}_frames")
    tmp_file = os.path.join(image_subdir, "base64frames.json")
    if not os.path.exists(image_subdir):
        os.makedirs(image_subdir, exist_ok=True)
        base64frames, _ = read_video(video_path, total_frames)
        for i, frame in enumerate(base64frames):
            with open(os.path.join(image_subdir, f"frame_{i}.jpg"), "wb") as f:
                f.write(base64.b64decode(frame))
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(base64frames, f)
    else:
        if os.path.exists(tmp_file):
            with open(tmp_file, "r", encoding="utf-8") as f:
                base64frames = json.load(f)
        else:
            base64frames = []
            for i in range(total_frames):
                with open(os.path.join(image_subdir, f"frame_{i}.jpg"), "rb") as f:
                    base64frames.append(base64.b64encode(f.read()).decode('utf-8'))
    
    return base64frames


def prepare_base64_video(video_url):
    video_path, video_id = download_video(video_url)

    video_base = base64.b64encode(open(video_path, "rb").read()).decode('utf-8')

    return video_base


def prepare_gemini_video_input(video_path):
    video_path, _ = download_video(video_path)

    # the dir of the video
    video_dir = os.path.dirname(video_path)
    if os.path.exists(os.path.join(video_dir, "gemini_video_cache.txt")):
        video_file_name = open(os.path.join(video_dir, "gemini_video_cache.txt"), "r").read().strip()
        try:
            video_file = genai.get_file(video_file_name)
        except:
            video_file = genai.upload_file(path=video_path)
            with open(os.path.join(video_dir, "gemini_video_cache.txt"), "w") as f:
                f.write(video_file.name)
    else:
        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(0.2)
            video_file = genai.get_file(video_file.name)
        with open(os.path.join(video_dir, "gemini_video_cache.txt"), "w") as f:
            f.write(video_file.name)
        
    return video_file

def read_video(video_path: str, total_frames: int):
    # Create a VideoCapture object
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file.")
    try:
        # Initialize a list to store base64 encoded frames
        base64_frames = []
        
        # Read frames in a loop
        while True:
            success, frame = video.read()
            if not success:
                break  # No more frames or error occurred

            # Encode the frame as a JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Convert the image to base64 string
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_base64)

        random.seed(42)
        if total_frames == 1:
            selected_indices = [np.random.choice(range(total_frames))]
        else:
            selected_indices = np.linspace(0, len(base64_frames) - 1, total_frames, dtype=int)

        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0

        return selected_base64_frames, duration
    finally:
        # Release the video capture object
        video.release()

def get_duration(video_path: str):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = int(frame_count / fps)
    return duration

def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    num_frames = min(num_frames, total_frames)

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames

def video_to_ndarrays(path: str, num_frames: int = -1) -> npt.NDArray:
    video_name = os.path.basename(path).split('.')[0]
    save_dir = os.path.join(os.path.dirname(path), f"{num_frames}_frames")
    npy_path = os.path.join(save_dir, "frames.npy")
    
    if os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        frames = np.stack(frames)
        frames = sample_frames_from_video(frames, num_frames)
        if len(frames) < num_frames:
            raise ValueError(f"Could not read enough frames from video file {path}"
                             f" (expected {num_frames} frames, got {len(frames)})")

        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)

        np.save(npy_path, frames)
        return frames

def video_to_ndarrays_fps(path: str, fps = 1, max_frames = 64) -> npt.NDArray:
    video_name = os.path.basename(path).split('.')[0]
    save_dir = os.path.join(os.path.dirname(path), f"{fps}fps_{max_frames}frames")
    npy_path = os.path.join(save_dir, "frames.npy")
    
    if os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        num_frames = fps * get_duration(path)
        num_frames = int(min(num_frames, max_frames))

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        frames = np.stack(frames)
        frames = sample_frames_from_video(frames, num_frames)
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)

        np.save(npy_path, frames)
        return frames
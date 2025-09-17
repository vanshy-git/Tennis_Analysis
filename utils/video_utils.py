import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames



def save_video(output_video_frames, output_video_path):
    # Ensure folder exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {output_video_path}")

    for frame in output_video_frames:
        out.write(frame)
    out.release()

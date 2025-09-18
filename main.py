from utils import (read_video, 
                   save_video)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector

def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)


    player_tracker = PlayerTracker(model_path='yolov5xu')
    ball_tracker = BallTracker(model_path='models/best.pt')
    player_detection = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="tracker_stubs/player_detections.pkl"
                                                    )

    ball_detection = ball_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                    stub_path="tracker_stubs/ball_detections.pkl"
                                                    )
    
    #court line detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])




      ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames= ball_tracker.draw_bboxes(video_frames, ball_detection)


    # Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)


    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
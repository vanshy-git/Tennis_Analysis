from utils import (read_video, 
                   save_video)
from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from court_line_detector.court_line_detector import CourtLineDetector
import cv2

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
    ball_detection = ball_tracker.interpolate_ball_positions(ball_detection)
    
    #court line detector
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #Choose Players
    player_detection = player_tracker.choose_and_filter_players(court_keypoints, player_detection)




      ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detection)
    output_video_frames= ball_tracker.draw_bboxes(video_frames, ball_detection)


    # Draw Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Draw frame number on top of the video
    for i in range(len(output_video_frames)):
        cv2.putText(output_video_frames[i], f'Frame: {i+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    save_video(video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
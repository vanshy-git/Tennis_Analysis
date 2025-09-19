from utils import (read_video, 
                   save_video,
                   measure_distance,
                   draw_player_stats,
                   convert_pixel_distance_to_meters
                   )
import constants
from trackers import PlayerTracker,BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import pandas as pd
from copy import deepcopy


def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='yolov5xu')
    ball_tracker = BallTracker(model_path='models/best.pt')

    player_detections = player_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl"
                                                     )
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl"
                                                     )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    
    # Court Line Detector model
    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # choose players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # MiniCourt
    mini_court = MiniCourt(video_frames[0]) 

    # Detect ball shots
    ball_shot_frames= ball_tracker.get_ball_shot_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, 
                                                                                                          ball_detections,
                                                                                                          court_keypoints)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    } ]
    
    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/24 # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters( distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters/ball_shot_time_in_seconds * 3.6

        # Debug: Check available players in this frame
        print(f"Frame {start_frame}: Available players: {list(player_mini_court_detections[start_frame].keys())}")
        print(f"Frame {end_frame}: Available players: {list(player_mini_court_detections[end_frame].keys())}")

        # player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_mini_court_detections[start_frame][1]))

        # Get all available player IDs and find the opponent
        available_players = list(player_positions.keys())
        opponent_player_ids = [pid for pid in available_players if pid != player_shot_ball]
        
        if not opponent_player_ids:
            print(f"Warning: No opponent player found in frame {start_frame}")
            continue
            
        # If multiple opponents (shouldn't happen in tennis), take the first one
        opponent_player_id = opponent_player_ids[0]
        
        print(f"Player who shot ball: {player_shot_ball}, Opponent: {opponent_player_id}")

        # Check if opponent exists in both frames
        if (opponent_player_id not in player_mini_court_detections[start_frame] or 
            opponent_player_id not in player_mini_court_detections[end_frame]):
            print(f"Warning: Opponent player {opponent_player_id} not found in both frames. Skipping.")
            continue

        # opponent player speed
        distance_covered_by_opponent_pixels = measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters( distance_covered_by_opponent_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()
                                                                           ) 

        speed_of_opponent = distance_covered_by_opponent_meters/ball_shot_time_in_seconds * 3.6

        # Map player IDs to standardized IDs (1 and 2) for stats
        # This ensures your stats structure works regardless of actual player IDs
        if len(available_players) >= 2:
            sorted_players = sorted(available_players)
            player_id_mapping = {sorted_players[i]: i+1 for i in range(len(sorted_players))}
            
            mapped_player_shot = player_id_mapping[player_shot_ball]
            mapped_opponent = player_id_mapping[opponent_player_id]
        else:
            print(f"Warning: Less than 2 players found in frame {start_frame}")
            continue

        current_player_stats= deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{mapped_player_shot}_number_of_shots'] += 1
        current_player_stats[f'player_{mapped_player_shot}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{mapped_player_shot}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{mapped_opponent}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{mapped_opponent}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Handle division by zero
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df.apply(
        lambda row: row['player_1_total_shot_speed']/row['player_1_number_of_shots'] 
        if row['player_1_number_of_shots'] > 0 else 0, axis=1)
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df.apply(
        lambda row: row['player_2_total_shot_speed']/row['player_2_number_of_shots'] 
        if row['player_2_number_of_shots'] > 0 else 0, axis=1)
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df.apply(
        lambda row: row['player_1_total_player_speed']/row['player_2_number_of_shots'] 
        if row['player_2_number_of_shots'] > 0 else 0, axis=1)
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df.apply(
        lambda row: row['player_2_total_player_speed']/row['player_1_number_of_shots'] 
        if row['player_1_number_of_shots'] > 0 else 0, axis=1)

    # Draw output
    ## Draw Player Bounding Boxes
    output_video_frames= player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames= ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Draw court Keypoints
    output_video_frames  = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections, color=(0,255,255))    

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    ## Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()
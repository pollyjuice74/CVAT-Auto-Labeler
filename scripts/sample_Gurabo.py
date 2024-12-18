import av
import os
import random


def sample_Gurabo(base_path, output_dir, sample_colonie_times=5, sample_files_times=1000):
    """
    Randomly selects a colony folder from the given base path and then randomly selects 
    .mp4 files within that folder. For each selected video, it extracts a specified 
    number of random frames.

    The total amount of extracted frames is: sample_colonie_times * sample_files_times.
    
    Parameters:
    base_path (str): Path to the base directory containing colony folders.
    output_dir (str): Directory where the extracted frames will be saved.
    sample_colonie_times (int): Number of colonies to sample from.
    sample_files_times (int): Number of frames to sample from each selected video.
    """
    colonies = [f'col{str(i).zfill(2)}' for i in range(1, 11)]

    for _ in range(sample_colonie_times):
        # Select a random colony
        random_colony = random.choice(colonies)
        colony_path = os.path.join(base_path, random_colony)
        
        # Get all .mp4 files in the selected colony folder
        mp4_files = [f for f in os.listdir(colony_path) if f.endswith('.mp4')]
        if not mp4_files:
            print(f"No video files found in {colony_path}")
            continue
        
        # Select a random .mp4 file
        selected_video = random.choice(mp4_files)
        video_path = os.path.join(colony_path, selected_video)
        
        print(f"Selected video: {video_path}")
        # Open the selected video and extract frames
        sample_frames_from_video(video_path, output_dir, sample_files_times, random_colony, selected_video)


def sample_frames_from_video(video_path, output_dir, sample_files_times, random_colony, selected_video):
    """
    Opens a video file and randomly extracts a specified number of frames. 
    The frames are saved as individual images in the output directory.
    
    Parameters:
    video_path (str): Path to the video file to extract frames from.
    output_dir (str): Directory where the extracted frames will be saved.
    sample_files_times (int): Number of frames to extract from the video.

    Output:
    -Each frame is saved with the naming format:
        {random_colony}_{selected_video}_{frame_index}.jpg
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    total_frames = video_stream.frames
    print(f"Total frames in video: {total_frames}")

    # Randomly select 'sample_files_times' frame indices to extract
    frame_indices = sorted( random.sample(range(total_frames),
                                          min(sample_files_times, total_frames)) )

    # Iterate through the video frames and extract only the selected ones
    for i, frame in enumerate(container.decode(video_stream)):
        if i in frame_indices:
            print(f"Extracting frame: {i}/{total_frames}")
            frame.to_image().save(os.path.join(output_dir, f'{random_colony}_{selected_video}_{i}.jpg'))
        if i >= max(frame_indices):
            break
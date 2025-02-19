import subprocess

def create_video_from_images(image_folder, output_video, framerate=10):
    # Construct the ffmpeg command
    # Filenames PA_CommonRoad/planner/Frenet/figs/KL0CrosS2fig_{timestep}.png
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', f'{image_folder}/t_%d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command)

# Example usage
create_video_from_images('figs', 'FrenetPlannerCrossing_KL5_4.mp4')
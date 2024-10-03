import subprocess

def create_video_from_images(image_folder, output_video, framerate=10):
    # Construct the ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', f'{image_folder}/fig_%d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    # Run the ffmpeg command
    subprocess.run(ffmpeg_command)

# Example usage
create_video_from_images('figs', 'CROSSING_KL5_NODeadlock.mp4')
import yt_dlp

# Define the YouTube URL
video_url = 'https://www.youtube.com/watch?v=D5-y1WuvYKc'

# Set download options
ydl_opts = {
    'format': 'bestaudio', 
    'outtmpl': './%(title)s.%(ext)s',  # Customize output path and filename
}

# Use yt-dlp to download the video
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    print("Download completed successfully!")
except Exception as e:
    print(f"An error occurred: {e}")



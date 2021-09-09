import cv2
from glob import glob
import os


video_dir = "/home/nttung/person-in-context/BPA-Net/video_demo"
video_paths = glob(f"{video_dir}/*")
for video_path in ("/home/nttung/person-in-context/BPA-Net/video_demo/kicking_penalty_cut_2.mp4",):
	# video_name = video_path.split("/")[-1]
	video_basename = os.path.basename(video_path)
	video_name = os.path.splitext(video_basename)[0]
	os.makedirs(f"{video_dir}/{video_name}", exist_ok=True)
	
	# print(video_path)
	# exit()
	vidcap = cv2.VideoCapture(video_path)
	success, image = vidcap.read()
	count = 0
	print(video_name)
	while success:
		cv2.imwrite(f"{video_dir}/{video_name}/{count:06d}.jpg", image)     # save frame as JPEG file      
		success, image = vidcap.read()
		print('Read a new frame: ', success)
		count += 1
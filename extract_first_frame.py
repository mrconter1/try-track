import cv2

video = cv2.VideoCapture('video.mp4')
success, frame = video.read()

if success:
    cv2.imwrite('first_frame.jpg', frame)
    print("First frame extracted and saved as 'first_frame.jpg'")
else:
    print("Failed to extract frame")

video.release()


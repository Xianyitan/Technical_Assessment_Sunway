from ultralytics import YOLO, solutions
import cv2

# read model
model = YOLO("yolov8n.pt")
# read video
cap = cv2.VideoCapture("C:\\Users\\User\\Desktop\\Sunway_Test\\Traffic_Analysis_YOLOV8\\traffic_video.mp4")
assert cap.isOpened(), "Error reading video file"
# get the information of weight, height and fps
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# define line for counter and classes
line_points = [(50, 500), (1230, 500)]

# Video writer
video_writer = cv2.VideoWriter("traffic_video_counting_output.avi", 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, 
                               (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
    

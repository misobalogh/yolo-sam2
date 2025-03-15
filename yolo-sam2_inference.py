import torch

from ultralytics import YOLO
from ultralytics import SAM

IMG_PATH = "heightMap.png"
RUN = "train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLO model
yolo_model = YOLO(f"runs/detect/{RUN}/weights/best.pt")
# Get bounding boxes from YOLO detection
yolo_output = yolo_model(IMG_PATH, conf=0.3)[0]

# Load SAM model
sam_ckpt = "sam2_b.pt"
sam_model = SAM(sam_ckpt)

# Extract bounding boxes from YOLO detection output and pass them to SAM model
boxes = yolo_output.boxes.xyxy
sam_output = sam_model(yolo_output.orig_img, bboxes=boxes, verbose=False, device=DEVICE, save=True)[0]

id2label = yolo_output.names
class_ids = yolo_output.boxes.cls.int().tolist()

sam_output_ids = {i: class_id for i, class_id in enumerate(class_ids)}
sam_output.names = {k: id2label[int(v)] for k,v in sam_output_ids.items()}

sam_output.show()
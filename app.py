import cv2
import numpy as np



def load_yolo_model(yolo_cfg, yolo_weights, yolo_names):
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    with open(yolo_names, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes


#
def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return class_ids, confidences, boxes, indexes


def calculate_calibration_factor(boxes, reference_index, real_width):
    x, y, w, h = boxes[reference_index]
    pixel_width = w
    calibration_factor = real_width / pixel_width
    return calibration_factor


def measure_objects(boxes, indexes, calibration_factor):
    measurements = []
    for i in indexes:
        x, y, w, h = boxes[i]
        width_real = w * calibration_factor
        height_real = h * calibration_factor
        measurements.append((width_real, height_real))
    return measurements


# Load yolov3 model
yolo_cfg = 'yolov3.cfg'
yolo_weights = 'yolov3.weights'
yolo_names = 'coco.names'
net, output_layers, classes = load_yolo_model(yolo_cfg, yolo_weights, yolo_names)

# load test image
img = cv2.imread('image.jpg')

# detect objects
class_ids, confidences, boxes, indexes = detect_objects(img, net, output_layers)

# reference object definition
reference_index = 0

# real object definition by human
real_width = 2.0

# calibration factor definition
calibration_factor = calculate_calibration_factor(boxes, reference_index, real_width)

# object measurement
measurements = measure_objects(boxes, indexes, calibration_factor)

# results
for i, measurement in enumerate(measurements):
    print(f"Object {i + 1}: Width = {measurement[0]} cm, Height = {measurement[1]} cm")

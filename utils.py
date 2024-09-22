import cv2

# Load YOLO model and get output layers
def load_model():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load YOLOv3 class labels
    with open("coco.names", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return net, output_layers, labels

# Perform object detection on an image
def detect_objects(img, model, output_layers, conf_threshold, nms_threshold):
    # Get image dimensions
    height, width = img.shape[:2]

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    model.setInput(blob)
    outputs = model.forward(output_layers)

    # Post-process the detection results
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                # Scale the bounding box back to the size of the image
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression to eliminate redundant overlapping boxes with lower confidence
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    detections = [(class_ids[i], confidences[i], boxes[i]) for i in indices.flatten()]

    return detections

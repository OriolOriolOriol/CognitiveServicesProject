import os,sys,time
import numpy as np
from colorama import Fore
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from cv2 import cv2

def detect(frame,detection_graph):
    
 
    sess = tf.Session(graph=detection_graph)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') # Defining tensors for the graph
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') # Each box denotes part of image with a person detected 
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') # Score represents the confidence for the detected person
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0') #Number of detections 


    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(frame, axis=0)
    # Pass the inputs and outputs to the session to get the results 
    (boxes, scores, classes, num) = sess.run([detection_boxes,detection_scores,detection_classes,num_detections],feed_dict={image_tensor: image_np_expanded})
    
    im_height, im_width,_ = frame.shape
    boxes_list = [None for i in range(boxes.shape[1])]
    for i in range(boxes.shape[1]):
        boxes_list[i] = (int(boxes[0,i,0] * im_height),int(boxes[0,i,1]*im_width),int(boxes[0,i,2] * im_height),int(boxes[0,i,3]*im_width))

    return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])


def detection(path,detection_graph,category_index,nome_video_risultato,numero_ground_truth_dataset_shanghai,conteggio_frame):
    frame=path
    numero_rilevazioni=0
    print(f"NUMERO GT: {numero_ground_truth_dataset_shanghai}")
    image=cv2.imread(frame)
    if image.size==0:
        sys.exit("L'immagine non è stata caricata correttamente")
    #print("OK caricata correttamente")
    boxes, scores, classes, num = detect(image,detection_graph)
    for i in range(len(boxes)):
        if classes[i]==1 and scores[i] > 0:
            numero_rilevazioni+=1
            box=boxes[i]
    
    print(f"NUMERO RILEVATI: {numero_rilevazioni}")

    with open("risultati_shagnhai.txt","a") as file:
        file.write(f"{frame}  {numero_ground_truth_dataset_shanghai} {numero_rilevazioni}")
        file.write("\n")


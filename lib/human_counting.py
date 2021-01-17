'''
Lo score indica quanto è un essere umano
boxes=100
con box ottento i counting box per ogni persona identificata nel frame

1) FP: false positive (non è una persona)
2) TP: true positive (è una persona)
3) FN: un ground truth che non è stato rilevato --> lo prendi dalla lista già creata "lista_persone"


'''
from lib.Evaluator import *
from collections import namedtuple
import os,glob,time,sys
from cv2 import cv2
import csv
import numpy as np
from colorama import Fore
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.python.util.deprecation as deprecation
import matplotlib.pyplot as plt


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
deprecation._PRINT_DEPRECATION_WARNINGS = False
evaluator = Evaluator()

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


 #LETTURA GROUND TRUTH DATASET KITTY  

def lettura_ground_truth(index):
    lista_coordinate=[]

    DIR_file="/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/dataset/KITTI-17/gt/gt.txt"
    data=dict()
    with open(DIR_file,"r") as file:
        for i,row in enumerate(file):
            row=row.split(",")
         
            if index == int(row[0]):
                #indica il frame!!!!
                frame=int(row[0])
         
                #prendo le coordinate di ogni ground truth presente nei vari frame
                row=row[2:6]
                for i in range(0, len(row)):
                    try: 
                        row[i] = int(row[i])
                    except ValueError:
                        row[i] = int(float(row[i]))

                #Trasformo le altre due coordinates
                row[2]=row[0] + row[2]
                row[3]=row[1] + row[3]
                #print(row)
                lista_coordinate.append(row)
    
    return lista_coordinate



def run_detection(DIR,detection_graph,category_index,nome_video_risultato):
    lista_precisione_totale=[]
    lista_recall_totale=[]
    sommaTP_FINALE=[]
    sommaFP_FINALE=[]
    sommaFN_FINALE=[]
    IOU_threshold=[0.5,0.75,0.90]
    for i in IOU_threshold:
        Soglia_iou=float(i)
        print(f"IOU {Soglia_iou}")
        frames=sorted(os.listdir(DIR))
        conteggio_frame=1

        TP=0
        FP=0
        FN=0

        somma_true_positive = 0
        somma_false_positive = 0
        somma_false_negative = 0

        lista_precisione=[]
        lista_recall=[]
        numero_totale_ground_truth=0
        
        start_time = time.time()
        for element in frames:
            lista_coordinate_modello_temp=[]
            frame=DIR+"/"+element
            image=cv2.imread(frame)
            if image.size==0:
                sys.exit("L'immagine non è stata caricata correttamente")
            
            boxes, scores, classes, num = detect(image,detection_graph)
            print(Fore.YELLOW+ "FRAME N." + str(conteggio_frame))
            lista_coordinate_gt=lettura_ground_truth(conteggio_frame)
            numero_ground_truth=len(lista_coordinate_gt)
            numero_totale_ground_truth+=numero_ground_truth

            print(Fore.RED + "")

            for i in range(len(boxes)):
                if classes[i]==1 and scores[i] > 0.5:
                    #TP=+1
                    box=boxes[i]
                    box_modello=(box[1],box[0],box[3],box[2])
                    lista_coordinate_modello_temp.append(box_modello)
            
            
            #Calcolo della IOU per ogni singolo frame
            for x in lista_coordinate_gt:
                x=tuple(x)
                for y in lista_coordinate_modello_temp:
                    iou=evaluator.iou(x,y)
                    if iou >= Soglia_iou:
                        TP+=1
                        print(f"Ground Truth: {str(x)}    Bounding Box: {str(y)}     IOU: {str(iou)}")
                        cv2.rectangle(image,(y[0],y[1]),(y[2],y[3]),(0,255,0),2)
                    elif iou > 0.2 and iou < Soglia_iou:
                        FP+=1



            detection=len(lista_coordinate_modello_temp)
            FN=(numero_ground_truth-TP)
            time.sleep(1)
            #varie valutazioni
            precisione_temp=evaluator.precision(int(TP),int(FP))
            recall_temp=evaluator.Recall(int(TP),int(FN))
            lista_precisione.append(precisione_temp)
            lista_recall.append(recall_temp)





            cv2.putText(image,'Detected = '+ str(detection),(10,100),cv2.FONT_HERSHEY_SIMPLEX, 1.25,(255,255,0),2,cv2.LINE_AA)
            cv2.imwrite("/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/Risultati/"+ nome_video_risultato +"/result%04i.jpg" %conteggio_frame,image)
            print(Fore.BLUE + f"Ground Truth: {str(numero_ground_truth)}   TP: {str(TP)}   FP: {str(FP)}  FN:{str(FN)}")
            #passo al prossimo frame
            conteggio_frame+=1
            somma_true_positive+=TP
            somma_false_negative+=FN
            somma_false_positive+=FP
            #Riazzero i parametri per il prossimo frame
            TP=0
            FP=0
            FN=0
        

        
       
        lista_precisione_totale.append(lista_precisione)
        sommaTP_FINALE.append(somma_true_positive)
        sommaFP_FINALE.append(somma_false_positive)
        sommaFN_FINALE.append(somma_false_negative)
        lista_recall_totale.append(lista_recall)
        conteggio_frame=0


    
    #CREAZIONE GRAFICO PRECISION X RECALL + SCRIVO SU FILE CSV RISULTATI OTTENUTI
    evaluator.grafico(nome_video_risultato,lista_precisione_totale,lista_recall_totale)
    tempo=(time.time() - start_time)
    for (a,b,c,d,e,f) in zip(lista_precisione_totale,lista_recall_totale,sommaTP_FINALE,sommaFP_FINALE,sommaFN_FINALE,IOU_threshold):
        media_precisione=evaluator.precision_Average(a)
        media_recall=evaluator.recall_Average(b)
        with open('/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/Risultati/risultati.csv',mode='a') as csvfile:
            nomicolonne = ['Model Name', 'Ground Truth(GT)', 'True Positive(TP)','False Positive(FP)','False Negative(FN)','Average Precision','Average Recall','Average Time']
            writer= csv.DictWriter(csvfile,fieldnames=nomicolonne)
            writer.writeheader()
            writer.writerow({ 'Model Name': f'{nome_video_risultato} con IOU = {f}', 'Ground Truth(GT)': numero_totale_ground_truth ,'True Positive(TP)': c,'False Positive(FP)': d,'False Negative(FN)':e,'Average Precision': media_precisione,'Average Recall':media_recall,'Average Time': tempo})

  
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





'''
Funzione che potrebbe non servire più
Errore da considerare:
Durante i vari frame il modello può intercettare più persone dei ground truth!? se ho TP=6 e FN=5 ovviamente mi andrà list of the range....
'''
def confronto(modello,ground_truth,index,index2,img):

   #try:
    #Lavoro per tirare fuori ogni singola coppia dei ground_truth e confrontarlo con i bounding box
    lista_ground_truth_temp=[]
    lista_ground_truth_frame=ground_truth[index]
    lista_ground_truth_frame=lista_ground_truth_frame.split("),")
    for i in lista_ground_truth_frame:
        rettangolo_ground_truth=str(i)
        if ");" in i:
            rettangolo_ground_truth=rettangolo_ground_truth.replace(");\n","")
        lista_ground_truth_temp.append(rettangolo_ground_truth)
    
    
    #Confronto tra la coppia del modello di quel particolare frame e la coppia ground truth di quel particolare frame
    ground_truth_now=lista_ground_truth_temp[index2]
    ground_truth_now1=[]
    #mi trasforma le coordinate di ogni singolob ox in array(Non so se mi può servire)
    #modello = np.asarray(modello)
    ground_truth_now=ground_truth_now.split(",")
    ground_truth_now1=[]
    for element in ground_truth_now:
        if "(" in element:
            element=element.replace("(","")
        ground_truth_now1.append(element)
    
    #trasformo le coordinate da stringhe a interi dei singoli box
    for i in range(0, len(ground_truth_now1)): 
        ground_truth_now1[i] = int(ground_truth_now1[i]) 



    modello=tuple(modello) #<-- classico quello che per ora va
    modello=np.array(modello)

    #ground_truth_now1= tuple(ground_truth_now1) <--# quello che per ora va
    ground_truth_now1=np.array(ground_truth_now1)
    print(f"MODELLO: {str(modello)} GROUND_TRUTH: {str(ground_truth_now1)}")
    idxs_true, idxs_pred,ious,labels=evaluator.match_bboxes(ground_truth_now1,modello)
    print(f"{str(idxs_pred)}   {str(idxs_true)}  {str(ious)}  {str(labels)}")
    time.sleep(5000)
    #print(f"Modello: {str(modello)} ---> Ground Truth: {str(ground_truth_now1)}")


    #cv2.rectangle(img, (ground_truth_now1[0],ground_truth_now1[1]),(ground_truth_now1[2],ground_truth_now1[3]), (255, 0, 0), 2)
    #cv2.rectangle(img,  (modello[0],modello[1]),(modello[2],modello[3]), (255, 0, 0), 2)
    #iou=evaluator.iou(ground_truth_now1,modello)
    #iou = evaluator.bb_intersection_over_union(ground_truth_now1, modello)
# cv2.putText(img, "IoU: {:.4f}".format(iou), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# cv2.imshow("Image", img)
# cv2.waitKey(0)
    #time.sleep(3)
    #return iou
   #except IndexError:
   #    pass
   

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
   
    frames=sorted(os.listdir(DIR))
    conteggio_frame=1
    IOU_threshold=0.90

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
            '''
            if classes[i] !=1 and scores[i]!=0:
                #FP+=1
                box1=boxes[i]
                box_modello1=(box1[1],box1[0],box1[3],box1[2])
                #cv2.rectangle(image,(box[1],box1[0]),(box1[3],box1[2]),(255,0,0),2)
            '''
           
        #Calcolo della IOU per ogni singolo frame
        for x in lista_coordinate_gt:
            x=tuple(x)
            for y in lista_coordinate_modello_temp:
                iou=evaluator.iou(x,y)
                if iou >= IOU_threshold:
                    TP+=1
                    print(f"Ground Truth: {str(x)}    Bounding Box: {str(y)}     IOU: {str(iou)}")
                    cv2.rectangle(image,(y[0],y[1]),(y[2],y[3]),(0,255,0),2)
                elif iou > 0.2 and iou < IOU_threshold:
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
    

    tempo=(time.time() - start_time)

    
    #print(str(lista_precisione))
    #print("\n\n\n")
    #print(str(lista_recall))
    #Provo a costruire il grafico
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(lista_recall,lista_precisione, label='Precision x Recall')
    #ax.fill_between(lista_recall,lista_precisione, 0,facecolor="orange",color='blue',alpha=0.2)          # Transparency of the fil
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='center left');
    fig.savefig(f'/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/Risultati/{nome_video_risultato}.png')   # save the figure to file
    plt.close(fig)    # close the figure

    media_precisione=evaluator.precision_Average(lista_precisione)
    media_recall=evaluator.recall_Average(lista_recall)


    print("Scrivo sul file csv il risultato...\n")

    with open('/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/Risultati/risultati.csv',mode='a') as csvfile:
        nomicolonne = ['Model Name', 'Ground Truth(GT)', 'True Positive(TP)','False Positive(FP)','False Negative(FN)','Average Precision','Average Recall','Average Time']
        writer= csv.DictWriter(csvfile,fieldnames=nomicolonne)
        writer.writeheader()
        writer.writerow({ 'Model Name': f'{nome_video_risultato}', 'Ground Truth(GT)': numero_totale_ground_truth ,'True Positive(TP)': somma_true_positive,'False Positive(FP)': somma_false_positive,'False Negative(FN)':somma_false_negative,'Average Precision': media_precisione,'Average Recall':media_recall,'Average Time': tempo})


  
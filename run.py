from glob import glob
import os,pathlib,time
from colorama import Fore
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
from lib import label_map_util
from lib import human_counting
from lib import human_counting_shanghai
from cv2 import cv2
import numpy as np #algebra lineare
import pandas as pd #data processing, CSV file I/O
import os
import PIL.Image as Image
import os
import scipy
from scipy.io import loadmat
from matplotlib import pyplot as plt


'''
def lettura_ground_truth(scelta):
  global file_ground_truth,file_ground_truth2
  if scelta==1:
      file_finale=file_ground_truth
  else:
      file_finale=file_ground_truth2
  persone=[]
  ground_truth=[]
  try:
      with open(file_finale,"r") as file:
          Lines = file.readlines()
          count=0
          for line in Lines:
             count+=1 
             line= line.replace(line[:28],"")
             ground_truth.append(line)
             lunghezza=len(line.split(')'))
             lunghezza=lunghezza-1
             persone.append(lunghezza)
         
          return persone,ground_truth         
  except IOError:
      print("Il file non e' accessibile")
'''

def load_ground_truth_SHANGAI(lista_immagini_shangai,lista_gt,detection_graph,category_index,nome_video_risultato):
    lista_immagini="/home/claudio/Scrivania/Dataset_Shangai/images_shangai/"
    listaGROUNDTRUTH="/home/claudio/Scrivania/Dataset_Shangai/Ground_truth_Shangai/"
    lista_ground_truth_shanghai_singola_immagine=[]
    conteggio_frame=1
    for IMMAGINE, GROUND_TRUTH in zip(lista_immagini_shangai, lista_gt):
        path=lista_immagini+IMMAGINE
        path1=listaGROUNDTRUTH+GROUND_TRUTH
        print(Fore.YELLOW+ "FRAME N." + str(conteggio_frame))
        print(Fore.RED)
        #CARICO IMMAGINE
        img = Image.open(path)
        img_matrix = np.array(img)
        print(f"DIMENSIONE IMMAGINE: {img_matrix.shape}")#dimensione immagine con 3 che rappresenta il colore

        #CARICO GT DI QUELLA SPECIFICA IMMAGINE
        mat = scipy.io.loadmat(path1)
        img_matrix_annotated = np.copy(img_matrix)#mostra le matrici dell immagine
        k = np.zeros((img_matrix.shape[0], img_matrix.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        print(f"DIMENSIONE GROUND TRUTH: {gt.shape}")#172 righe 2 colonne per il primo elemento



        for i in range(0, len(gt)):
            #print("SHAPE 0: ", img_matrix_annotated.shape[0])#dimensione immagine altezza
            #print("SHAPE 1: ", img_matrix_annotated.shape[1])#dimensione immagine larghezza
            if int(gt[i][1]) < img_matrix_annotated.shape[0] and int(gt[i][0]) < img_matrix_annotated.shape[1]:
                #stampo le 4 annotazioni
                #print(img_matrix_annotated[int(gt[i][1]), int(gt[i][0]), 0])
                #print(img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]), 0])
                #print(img_matrix_annotated[int(gt[i][1]), int(gt[i][0]+1), 0] )
                #print(img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]+1), 0])
                #img_matrix_annotated[int(gt[i][1]), int(gt[i][0]), 0] = 255 # annotated point
                # make the point on figure bigger for visual 
                #img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]), 0] = 255 # 
                #img_matrix_annotated[int(gt[i][1]), int(gt[i][0]+1), 0] = 255 # 
                #img_matrix_annotated[int(gt[i][1]+1), int(gt[i][0]+1), 0] = 255 # 

                # Center coordinates
                center_coordinates = (int(gt[i][0]), int(gt[i][1]))
                lista_ground_truth_shanghai_singola_immagine.append(center_coordinates)
                numero_ground_truth_dataset_shanghai=len(lista_ground_truth_shanghai_singola_immagine)
                #print(lista_ground_truth_shanghai)


        #ESEGUO IL MODELLO
        human_counting_shanghai.detection(path,detection_graph,category_index,nome_video_risultato,numero_ground_truth_dataset_shanghai,conteggio_frame)
        conteggio_frame+=1






def load_model(modello,label):
    # Declare detection graph
    detection_graph = tf.Graph()
     # Load the model into the tensorflow graph
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(modello, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    

    #Create a session from the detection graph
    default_graph = detection_graph.as_default()
    num_classes = 90
    path_to_labels = os.path.join('data', label)

    #Labellare i rettangoli
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return detection_graph, category_index


#DATASET KITTY
DIR="/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/dataset/KITTI-17/img1"
#lista_persone,lista_ground_truth=lettura_ground_truth(scelta)

#LISTA MODELLI
lista=glob("/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/models/*")


#DATASET SHANGHAI
#lista_immagini_shangai="/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/dataset/Dataset_Shangai/images_shangai"
#lista_gt="/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/dataset/Dataset_Shangai/Ground_truth_Shangai"
#lista_immagini_shangai=sorted(os.listdir(lista_immagini_shangai))
#lista_gt=sorted(os.listdir(lista_gt))

for list1 in lista:
    #CARICAMENTO DEI VARI MODELLI DA ESEGUIRE
    existGDBPath = pathlib.Path(list1)
    path=os.path.split(os.path.abspath(existGDBPath))
    nome_video_risultato=path[1]
    print(Fore.GREEN + f"Starting with {Fore.RED} {nome_video_risultato}...")
    time.sleep(1)
    frozen=list1+ "/frozen_inference_graph.pb"
    #caricare il modello
    detection_graph, category_index = load_model(frozen, 'label.txt')
    print(Fore.GREEN + "Modello Caricato")
    #inizio della rilevazione
    print(Fore.RED)

   ###AVVIO SHANGHAI######################################################
    #with open("risultati_shagnhai.txt","a") as file:
    #    file.write(f"NOME MODELLO: {nome_video_risultato}")
    #    file.write("\n")
    
    #load_ground_truth_SHANGAI(lista_immagini_shangai,lista_gt,detection_graph,category_index,nome_video_risultato)
   

    #AVVIO DATASET KITTY
    human_counting.run_detection(DIR,detection_graph,category_index,nome_video_risultato)

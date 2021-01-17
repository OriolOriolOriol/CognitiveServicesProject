from __future__ import division
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt

class Evaluator:

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)


    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    def precision(self,TP,FP):
        try:
            precisione = TP/(TP+FP)
            if precisione > 1.0:
                precisione=1.0
            
        except ZeroDivisionError:
            precisione=0

        return precisione

    
    def Recall(self,TP,FN):
        try:
            recall= TP/(TP+FN)
            if recall > 1.0:
                recall=1.0
        except ZeroDivisionError:
            recall=0
            
        return recall

    
    def precision_Average(self,list_precision):
        somma=0
        for i in list_precision:
            somma+=i
        
        average_precision= somma/len(list_precision)
        return average_precision

    
    def recall_Average(self,list_recall):
        somma=0
        for i in list_recall:
            somma+=i
        
        average_precision= somma/len(list_recall)
        return average_precision
    

    def iou_average(self,list_iou):
        somma=0
        for i in list_iou:
            somma+=i
        
        average_precision= somma/len(list_iou)
        return average_precision


    #Secondo modo per calcolare la iou!!!!
    def bb_intersection_over_union(self,boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = float(interArea) / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return float(iou)
    

    #Terzo modo per calcolare la iou!!!!!
    def bbox_iou(self,boxA, boxB):
        
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = xB - xA + 1
        interH = yB - yA + 1

        # Correction: reject non-overlapping boxes
        if interW <=0 or interH <=0 :
            return -1.0

        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    """
    Calculate the AP given the recall and precision array
        1st) We compute a version of the measured precision/recall curve with
            precision monotonically decreasing
        2nd) We compute the AP as the area under this curve by numerical integration.
    """

    @staticmethod
    def voc_ap(rec, prec):
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

    
    @staticmethod
    def grafico(nome,listaPrecisioneCompleta,listaRecallCompleta):

        lista_precision_0_5=sorted(listaPrecisioneCompleta[0])[::-1]
        lista_recall_0_5=sorted(listaRecallCompleta[0])

        lista_precision_0_75=sorted(listaPrecisioneCompleta[1])[::-1]
        lista_recall_0_75=sorted(listaRecallCompleta[1])

        lista_precision_0_9=sorted(listaPrecisioneCompleta[2])[::-1]
        lista_recall_0_9=sorted(listaRecallCompleta[2])


        ap,mrec,mpre= Evaluator.voc_ap(lista_recall_0_5,lista_precision_0_5)
        ap1,mrec1,mpre1= Evaluator.voc_ap(lista_recall_0_75,lista_precision_0_75)
        ap2,mrec2,mpre2= Evaluator.voc_ap(lista_recall_0_9,lista_precision_0_9)

        print("\nCALCOLO AP..\n")
        print(f"{nome} con IOU 0.5 AP {ap}")
        print(f"{nome} con IOU 0.75 AP {ap1}")
        print(f"{nome} con IOU 0.90 AP {ap2}")

        if "inception" in nome:
            nome_finale="SSD Inception"
        elif "mask" in nome:
            nome_finale="MASK RCNN"
        
        elif "faster" in nome:
            nome_finale="FASTER RCNN"

        plt.plot(lista_recall_0_5,lista_precision_0_5,linewidth=1,color='blue',label = "IOU 0.50")
        plt.plot(lista_recall_0_75,lista_precision_0_75,linewidth=1,color='red',label = "IOU 0.75")
        plt.axhline(y=0, color="black", linestyle='-', label="NO Skills")
        plt.plot(lista_recall_0_9,lista_precision_0_9,linewidth=1,color='green',label = "IOU 0.90")
        plt.legend()
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.title(f'{nome_finale} Precision x Recall  IOU = {0.5 ,0.75, 0.90}')
        plt.savefig(f'/home/claudio/Scrivania/COGNITIVE_SERVICES_3.0/Risultati/{nome_finale}.png')

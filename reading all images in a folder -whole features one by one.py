# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:45:29 2022

@author: Admin
"""
import SimpleITK as sitk
import glob
import os
import SimpleITK as sitk
import six
import numpy
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D
import matplotlib.pyplot as plt
import pandas as pd
from radiomics import featureextractor, getTestCase

dataDir = 'F:/@Software/pyradiomics-master/pyradiomics-master'
parForder = os.path.join(dataDir, "examples", "exampleSettings", "Params-first-order.yaml")
parGLCM = os.path.join(dataDir, "examples", "exampleSettings", "Params-GLCM.yaml")
parGLDM = os.path.join(dataDir, "examples", "exampleSettings", "Params-GLDM.yaml")
parGLRLM = os.path.join(dataDir, "examples", "exampleSettings", "Params-GLRLM.yaml")
parGLSZM = os.path.join(dataDir, "examples", "exampleSettings", "Params-GLSZM.yaml")
parNGTDM = os.path.join(dataDir, "examples", "exampleSettings", "Params-NGTDM.yaml")

#for maskk in glob.glob("E:/99-CT/GLCM/segmentation data/dehghani/hypo-MHA/*.mha"):
  # maskim = sitk.ReadImage(maskk)
   
#for imgg in glob.glob("E:/99-CT/GLCM/segmentation data/dehghani/whole-MHA/*.mha"):
 #  image = sitk.ReadImage(imgg)
       
    
pathnorm = "E:/99-CT/GLCM/segmentation data/Normals/Yaminifar/R-MHA/*.mha"
norm=glob.glob(pathnorm)

pathhypo = "E:/99-CT/GLCM/segmentation data/Normals/Yaminifar/L-MHA/*.mha"
hypo=glob.glob(pathhypo)

pathwhole = "E:/99-CT/GLCM/segmentation data/Normals/Yaminifar/whole-MHA/*.mha"
whole=glob.glob(pathwhole)  

extractorFOrder = featureextractor.RadiomicsFeatureExtractor(parForder)  
extractorGLCM = featureextractor.RadiomicsFeatureExtractor(parGLCM) 
extractorGLDM = featureextractor.RadiomicsFeatureExtractor(parGLDM) 
extractorGLRLM = featureextractor.RadiomicsFeatureExtractor(parGLRLM) 
extractorGLSZM = featureextractor.RadiomicsFeatureExtractor(parGLSZM) 
extractorNGTDM = featureextractor.RadiomicsFeatureExtractor(parNGTDM) 
'''
settings={}
settings['binWidth']=25
settings['resampledPixelSpacing']=[3,3,3]
settings['interpolator']='stikBSpline'
settings['verbose']=True
settings['force2D']=True
settings['force2Ddimension']=0
settings['label'] = 1
'''
reshypoForder=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorFOrder.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypoForder.append(whole_results)
reshypoForder

import csv
#converting list of dictionary to CSV 
keys = reshypoForder[0].keys()

with open('first-order-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypoForder)

resnormForder=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorFOrder.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormForder.append(whole_results)
resnormForder


keys = resnormForder[0].keys()

with open('first-order-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormForder)
    
#glcm

reshypoglcm=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLCM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypoglcm.append(whole_results)
reshypoglcm

import csv
#converting list of dictionary to CSV 
keys = reshypoglcm[0].keys()

with open('glcm-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypoglcm)

resnormglcm=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLCM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormglcm.append(whole_results)
resnormglcm


keys = resnormglcm[0].keys()

with open('glcm-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormglcm)

#gldm

reshypogldm=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLDM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypogldm.append(whole_results)
reshypogldm

import csv
#converting list of dictionary to CSV 
keys = reshypogldm[0].keys()

with open('gldm-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypogldm)

resnormgldm=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLDM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormgldm.append(whole_results)
resnormgldm


keys = resnormgldm[0].keys()

with open('gldm-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormgldm)
    
#GLRLM  
  
reshypoglrlm=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLRLM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypoglrlm.append(whole_results)
reshypoglrlm

import csv
#converting list of dictionary to CSV 
keys = reshypoglrlm[0].keys()

with open('glrlm-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypoglrlm)

resnormglrlm=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLRLM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormglrlm.append(whole_results)
resnormglrlm


keys = resnormglrlm[0].keys()

with open('glrlm-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormglrlm)

#GLSZM    
reshypoglszm=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLSZM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypoglszm.append(whole_results)
reshypoglszm

import csv
#converting list of dictionary to CSV 
keys = reshypoglszm[0].keys()

with open('glszm-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypoglszm)

resnormglszm=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorGLSZM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormglszm.append(whole_results)
resnormglszm


keys = resnormglszm[0].keys()

with open('glszm-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormglszm)    
    
#NGTDM    

reshypongtdm=[]
for msk,im in zip(hypo,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorNGTDM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      reshypongtdm.append(whole_results)
reshypongtdm

import csv
#converting list of dictionary to CSV 
keys = reshypongtdm[0].keys()

with open('ngtdm-Yaminifar-L.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(reshypongtdm)

resnormngtdm=[]
for msk,im in zip(norm,whole):
      mask = sitk.ReadImage(msk)
#image3d = sitk.JoinSeries(image)
      image = sitk.ReadImage(im)
#mask3d = sitk.JoinSeries(mask)
      whole_results=extractorNGTDM.execute(image,mask)            
      #glcmFeatures = glcm.RadiomicsGLCM(image, mask, **settings)
      #glcmFeatures.enableAllFeatures()
      #results = glcmFeatures.execute()
      resnormngtdm.append(whole_results)
resnormngtdm


keys = resnormngtdm[0].keys()

with open('ngtdm-Yaminifar-R.csv', 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(resnormngtdm)    
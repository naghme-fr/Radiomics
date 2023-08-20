from radiomics import featureextractor
import numpy
import SimpleITK as sitk
from radiomics import featureextractor
from radiomics import firstorder, getTestCase, glcm, glrlm, glszm, imageoperations, shape, shape2D
import six
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
#from plt import volview


#imageName, maskName = getTestCase('lung2')

#im_tiff = sitk.ReadImage('D:/Dr.Ghasem zadeh/Sample 133-20211113T100454Z-001/Sample 133/133_60kv_25_10_rec_voi_Metha00002612.bmp')
#im_vect = sitk.JoinSeries(im_tiff)  # Add 3rd dimension,NO LONGER NECESSARY
#im = sitk.VectorImageSelectionCast(im_vect, 0, sitk.sitkFloat64)
imageName = 'E:/99-CT/features/segmentationdata/dehghani/whole.nrrd'
maskName = 'E:/99-CT/features/segmentationdata/dehghani/hypo.nrrd'

#2D
imageName = 'E:/99-CT/features/segmentationdata/dehghani/whole-MHA/Dehghani-Amir062.tif.mha'
maskName = 'E:/99-CT/features/segmentationdata/dehghani/hypo-MHA/hypo062.tif.mha'

image = sitk.ReadImage(imageName)
#image3d = sitk.JoinSeries(image)
mask = sitk.ReadImage(maskName)
#mask3d = sitk.JoinSeries(mask)

#im3d=sitk.GetArrayViewFromImage(image3d)
#im3d_3d=im3d[0,:,:,:]#3 channel shod  #chon baraye 4d engar error midad..
#im3=sitk.GetImageFromArray(im3d_3d)

#msk3d=sitk.GetArrayViewFromImage(mask3d)
#msk3d_3d=msk3d[0,:,:,:]
#mask3=sitk.GetImageFromArray(msk3d_3d)
#mask3d.CopyInformation(im3)

repo_root = r'F:/@Software/pyradiomics-master/pyradiomics-master'  # Update this variable

#dataDir = os.path.join(repo_root , 'data')
examples_settings_dir = os.path.join(repo_root, 'examples', 'exampleSettings')

#imageName, maskName = getTestCase('brain1', dataDir)
params = os.path.join(examples_settings_dir, "exampleVoxelsad.yaml") #chon memory error dad batch ro az 10000 be 100 taghir dadam

extractor = featureextractor.RadiomicsFeatureExtractor(params)

#extractor.disableAllFeatures()  # disable all features
#extractor.enableFeaturesByName(glcm=['ClusterProminence'])  # Only enable firstorder mean

#extractor.settings['initValue'] = 0  # Set the non-calculated voxels to 0

result = extractor.execute(image, mask,voxelBased=True) #label ro khodam ezafe kardam 

for key, val in six.iteritems(result):
  if isinstance(val, sitk.Image):
    # Do something to the featuer, e.g.
    parametermap = sitk.GetArrayFromImage(val)
  else:
    # Diagnostic feature, just print the value for now
    print('%s: %s' % (key, str(val)))
    
#for showing 3D     
#plt.imshow(parametermap[24, :, :])
#plt.show()

#for showing 2D 
plt.imshow(parametermap)
plt.show()
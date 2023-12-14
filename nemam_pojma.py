from datetime import datetime
start_time = datetime.now()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
from keras.layers import ELU, PReLU, LeakyReLU
from keras.layers import Conv2D, Conv2DTranspose, Dense, Input, Reshape, Flatten,Lambda
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.datasets import mnist
from keras.callbacks import  EarlyStopping
import math as m
from matplotlib import pyplot as plt
from matplotlib import ticker
import mne
from mne.io import read_raw_bdf
import numpy as np
import pandas as pd
import pdb
# import plotly.plotly as py
import random
import scipy.io
from scipy.io import loadmat
from scipy.interpolate import griddata
from skimage import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from calculate_bdm_old import calculate_bdm



def readDataset(verbose=False, path='s01.bdf'):

  raw = mne.io.read_raw_bdf(path, preload=True)
  #print(raw)
  #print(raw.info)
  #print(raw._data.shape)
  rawDataset = raw.copy()
  ch_names= ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']#, 'Status']
  rawDataset=rawDataset.drop_channels(rawDataset.ch_names[47:], on_missing='ignore')
  rawDataset=rawDataset.drop_channels(ch_names, on_missing='raise')
  
  #rawDatasetForMontageLocation=rawDataset.copy()

  channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
          'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
          'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']

  montage = mne.channels.make_standard_montage('biosemi32')


  info = mne.create_info(ch_names=channelNames, sfreq=128, ch_types='eeg')
  info.set_montage(montage)


  #A bandpass frequency filter from 4.0-45.0Hz was applied.
  rawDataset.filter(0.1, 50., fir_design='firwin')


  #pre-process data
  #downsampling to 128
  rawDataset = rawDataset.resample(sfreq=128)

  # use the average of all channels as reference
  rawDatasetReReferenced = rawDataset.copy().set_eeg_reference(ref_channels='average')
  rawDatasetForMontageLocation=rawDatasetReReferenced.copy()



  transposedDataset = np.transpose(rawDatasetReReferenced._data)
  print ("Length of data: {}".format(len(transposedDataset)))


  if (verbose):
    rawDataset.plot()
    rawDatasetReReferenced.plot()

  if (verbose):
    for i in range(10):
      randomIndex = random.randint(0,len(transposedDataset))
      print ("Random index: {}".format(randomIndex))
      fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5,3), gridspec_kw = {'wspace':0, 'hspace':0.2})
      sampleUnderAnalysis = transposedDataset[randomIndex,:]
      #print (sampleUnderAnalysis)

      ax = axes
      ax.title.set_text('raw dataset')
      im,_ = mne.viz.topomap.plot_topomap(sampleUnderAnalysis,info,names=channelNames,axes=ax,cmap='Spectral_r',show=False)
      #ax.axis('off')

      plt.show()

  return transposedDataset,rawDatasetForMontageLocation


def getChannelNames():
  channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
        'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
        'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
  return channelNames

def getChannellInfoForSample (channelNames, channelValues, onlyValues=False):
  i = 0
  channelValuesforCurrentSample = []
  for ch in channelNames:
    chValue = channelValues[i]
    if (onlyValues):
      channelValuesforCurrentSample.append(chValue)
    else:
      channelValuesforCurrentSample.append((ch, chValue))
    i+=1

  return channelValuesforCurrentSample


def azim_proj(pos):
  """
  Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
  Imagine a plane being placed against (tangent to) a globe. If
  a light source inside the globe projects the graticule onto
  the plane the result would be a planar, or azimuthal, map
  projection.

  :param pos: position in 3D Cartesian coordinates    [x, y, z]
  :return: projected coordinates using Azimuthal Equidistant Projection
  """
  [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
  return pol2cart(az, m.pi / 2 - elev)


def cart2sph(x, y, z):
  """
  Transform Cartesian coordinates to spherical
  :param x: X coordinate
  :param y: Y coordinate
  :param z: Z coordinate
  :return: radius, elevation, azimuth
  """
  x2_y2 = x**2 + y**2
  r = m.sqrt(x2_y2 + z**2)                    # r     tant^(-1)(y/x)
  elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
  az = m.atan2(y, x)                          # Azimuth
  return r, elev, az


def pol2cart(theta, rho):
  """
  Transform polar coordinates to Cartesian
  :param theta: angle value
  :param rho: radius value
  :return: X, Y
  """
  #print ('----------')
  #print (rho * m.cos(theta))
  #print (rho * m.sin(theta))
  return rho * m.cos(theta), rho * m.sin(theta)


def getBrainRateForTimeSlice(brainRatesForChannels):
  #return (np.mean(brainRatesForChannels))
  return (np.sum(brainRatesForChannels))

def get3DCoordinates(MontageChannelLocation,EEGChannels):
  MontageChannelLocation=MontageChannelLocation[-EEGChannels:]
  location=[]
  for i in range(0,32):

    v=MontageChannelLocation[i].values()
    values = list(v)
    a=(values[1])*1000
    location.append(a)
  MontageLocation=np.array(location)
  #MontageLocation=trunc(MontageLocation,decs=3)
  MontageLocation= np.round(MontageLocation,1)
  MontageLocation=MontageLocation.tolist()
  return MontageLocation


def convert3DTo2D(pos_3d):
  pos_2d = []
  for e in pos_3d:
    pos_2d.append(azim_proj(e))
  return (pos_2d)

#Function to Compute SSIM, MSE, and MAE
def SSIM(actual,pred):
  ssimErrAllPair = 0
  for i in range(actual.shape[0]):
      ssimErrAllPair += metrics.structural_similarity(actual[i], pred[i].astype('float64'), data_range=1)/ actual.shape[0]
  return ssimErrAllPair

def MSE(actual,pred):
  mseErrAllPair = 0
  for i in range(actual.shape[0]):
      mseErrAllPair += metrics.mean_squared_error(actual[i], pred[i]) / actual.shape[0]
  return mseErrAllPair

def MAE(actual,pred):
  MA_error_all = 0
  for i in range(actual.shape[0]):
      MA_error_all += mean_absolute_error(actual[i], pred[i].astype('float64'))/ actual.shape[0]
  return MA_error_all

def MAPE(actual,pred):
  from sklearn.metrics import mean_absolute_percentage_error
  MA_perror_all = 0
  # maximum=-10000000
  # maxInd=0
  for i in range(actual.shape[0]):
      temp=mean_absolute_percentage_error(actual[i], pred[i].astype('float16'))
      MA_perror_all += temp/ actual.shape[0]
  #     if(temp>maximum):
  #       maximum=temp
  #       maxInd=i
  # print(maximum, maxInd)
  return MA_perror_all



# Create topographic map for a specific sample (of 32 channels)
def createTopographicMapFromChannelValues(channelValues, interpolationMethod = "cubic", verbose=False):
  #retrieve the names of channels
  channelNames = getChannelNames()
  #channelValues=channelValues.transpose(1,0)
  listOfChannelValues = getChannellInfoForSample(channelNames, channelValues, onlyValues=True)
  #print(listOfChannelValues)
  #create an empty (with zeros) topographic map of lengthOfTopographicMap x lengthOfTopographicMap pixels
  emptyTopographicMap = np.array(np.zeros([lengthOfTopographicMap, lengthOfTopographicMap]))
  if (verbose):
    #print the empty topographic map
    plt.imshow(emptyTopographicMap)
    plt.show()

  pos2D=np.array(convert3DTo2D(get3DCoordinates(MontageChannelLocation,NumberOfEEGChannel)))
  #print(pos2D)
  #input()
  grid_x, grid_y = np.mgrid[
                     min(pos2D[:, 0]):max( pos2D[:, 0]):lengthOfTopographicMap*1j,
                     min( pos2D[:, 1]):max( pos2D[:, 1]):lengthOfTopographicMap*1j
                     ]
  #print(grid_x)
  #print(grid_y)
  #Generate edgeless images
  min_x, min_y = np.min(pos2D, axis=0)
  max_x, max_y = np.max(pos2D, axis=0)
  locations = np.append(pos2D, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
  #channelValuesForSample =channelValues
  #print(locations)
  #input()
  # interpolate edgeless topographic map and show it
  interpolatedTopographicMap = griddata(pos2D, channelValues, (grid_x, grid_y), method=interpolationMethod, fill_value=0)


  #coordinates where the pixel values are zero
  CordinateYellowRegion=np.argwhere(interpolatedTopographicMap == 0.)

  if (verbose):
    i=0
    for chVal in channelValues:
      for x in range (32):
        for y in range (32):
          print ("Trying to find value {} in pixel ({},{}-{})".format(chVal, x, y, interpolatedTopographicMap[x][y] ))
          if (chVal == interpolatedTopographicMap[x][y]):
            print ("Value found at pixel ({}{}) for channel: {}".format(x, y,channelNames[i] ))
      i=i+1

  if (verbose):
    plt.imshow(interpolatedTopographicMap)
    plt.show()

  return interpolatedTopographicMap,CordinateYellowRegion

def removeInterpolation(interpolatedTopographicMap):
  pos2D = np.array(convert3DTo2D(get3DCoordinates(MontageChannelLocation, NumberOfEEGChannel)))
  emptyTopographicMap = np.array(np.zeros([lengthOfTopographicMap, lengthOfTopographicMap]))

  # Map interpolated values to the closest pixel locations
  for i in range(len(pos2D)):
    x, y = pos2D[i]
    x_idx = int(round((x - min(pos2D[:, 0])) / (max(pos2D[:, 0]) - min(pos2D[:, 0])) * (lengthOfTopographicMap - 1)))
    y_idx = int(round((y - min(pos2D[:, 1])) / (max(pos2D[:, 1]) - min(pos2D[:, 1])) * (lengthOfTopographicMap - 1)))
    
    emptyTopographicMap[y_idx, x_idx] = interpolatedTopographicMap[y_idx, x_idx]

  return emptyTopographicMap


#initialisation for all required variables.
sfreq=128
NumberOfEEGChannel=32
lengthOfTopographicMap=40
StartingSamplePoint= 16905
#EndSamplePoint= rawDataset.shape[0]-1
EndSamplePoint= 20000
NumberOfTopomapsGenerating=EndSamplePoint-StartingSamplePoint
z_dim=32
epoch = 5
batch_size = 32
encoder_input_shape = (40, 40, 1)




#Design the encoder
encoder_input = Input(shape=encoder_input_shape) # It was "shape=(raw.shape[1], raw.shape[2],1)" but changed to this because it is always 40, 40 N.M.
en_conv1 = Conv2D(filters = 32, kernel_size=4, strides=2, padding='same')(encoder_input)
en_conv1 = LeakyReLU(0.1)(en_conv1)

en_conv2 = Conv2D(filters = 64, kernel_size=4, strides=2, padding='same')(en_conv1)
en_conv2 = LeakyReLU(0.1)(en_conv2)

en_conv3 = Conv2D(filters = 128, kernel_size=4, strides=2, padding='same')(en_conv2)
en_conv3 = LeakyReLU(0.1)(en_conv3)

#en_conv4 = Conv2D(filters = 256, kernel_size=4, strides=2, padding='same')(en_conv3)
#en_conv4 = LeakyReLU(0.1)(en_conv4)

en_fc1= Flatten()(en_conv3)
mu = Dense(z_dim)(en_fc1)
sigma = Dense(z_dim)(en_fc1)

#Compute latent using mean and variance
def compute_latent(x):
  mu, sigma = x
  batch = K.shape(mu)[0]
  dim = K.int_shape(mu)[1]
  eps = K.random_normal(shape=(batch,z_dim),mean=0., stddev=1.)
  return mu + K.exp(sigma)*eps
latent_space = Lambda(compute_latent)([mu, sigma])

#Build the encoder
encoder = Model(encoder_input,[mu,sigma,latent_space], name='encoder')
#plot_model(encoder, show_shapes=True, to_file='encoder_model_plot.png')
encoder.summary()



#Design the decoder'
decoder_input = Input(shape=(z_dim,))
de_fc2 = Dense(en_conv3.shape[1]*en_conv3.shape[2]*en_conv3.shape[3], activation='relu')(decoder_input )
de_fc2 = Reshape((en_conv3.shape[1], en_conv3.shape[2],en_conv3.shape[3]))(de_fc2)
de_conv1 = Conv2DTranspose(filters=128, kernel_size=4, strides=2, activation='relu', padding='same')(de_fc2)
de_conv2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, activation='relu', padding='same')(de_conv1)
#de_conv3 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, activation='relu', padding='same')(de_conv2)
decoder_output = Conv2DTranspose(filters=1, kernel_size=4, strides=2, activation='sigmoid', padding='same')(de_conv2)

decoder = Model(decoder_input, decoder_output, name='decoder')
#plot_model(decoder, show_shapes=True, to_file='decoder_model_plot.png')
decoder.summary()

#Build the autoencoder for training
train_z = encoder(encoder_input)[2]
train_xr = decoder(train_z)
autoencoder = Model(encoder_input, train_xr)
autoencoder.summary()
#plot_model(autoencoder, show_shapes=True)


#MMD functions
def compute_kernel(x, y):
  x_size = K.shape(x)[0]
  y_size = K.shape(y)[0]
  dim = K.shape(x)[1]
  tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
  tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
  return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))

def compute_mmd(x, y):
  x_kernel = compute_kernel(x, x)
  y_kernel = compute_kernel(y, y)
  xy_kernel = compute_kernel(x, y)
  return K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

#function for loss function
def custom_loss(train_z, train_xr, train_x):
  loss_mmd = compute_mmd(latent_space, train_z)

  'Then, also get the reconstructed loss'
  loss_nll = K.mean(K.square(train_xr - train_x))

  'Add them together, then you can get the final loss'
  loss = loss_nll + loss_mmd
  return loss

#compile the model, prepare for training
loss = custom_loss(train_z, train_xr, encoder_input)
autoencoder.add_loss(loss)
autoencoder.compile(optimizer='adam')
myCallBacks = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='min')


data_normalized = []
#training the autoencoder
for person in range (1,2):
  #read dataset and generate Topographic Maps to train the autoencoder with listOfTopographicMapsForAE
  rawDataset,rawDatasetForMontageLocation= readDataset(verbose=False, path="s{:02d}.bdf".format(person))

  rawDatasetForMontageLocation.set_montage('biosemi32')
  MontageChannelLocation=rawDatasetForMontageLocation.info['dig']

  #extract the values of those pixels at the 32 coordinates from an interpolated topographic map
  #coordinates2D = get2DTopographicMapChannelIndexes(lengthOfTopographicMap,CordinateYellowRegion)
  channelNames = getChannelNames()

  listOfTopographicMapsForAE = []
  for i in range(StartingSamplePoint,EndSamplePoint):
    #get a random sample and load its 32 values
    channelValuesForCurrentSample = list(rawDataset[i,:])

    #create topographic map
    interpolatedTopographicMap,CordinateYellowRegion = createTopographicMapFromChannelValues(channelValuesForCurrentSample, interpolationMethod="cubic",verbose=False)
    #plt.imshow(interpolatedTopographicMap)
    #plt.show()
    listOfTopographicMapsForAE.append(interpolatedTopographicMap)
  
  listOfTopographicMapsForAE = np.array(listOfTopographicMapsForAE)

  dataNorm = (listOfTopographicMapsForAE - listOfTopographicMapsForAE.min()) / (listOfTopographicMapsForAE.max() - listOfTopographicMapsForAE.min())

  dataNormReshape=dataNorm.reshape(-1,listOfTopographicMapsForAE.shape[1],listOfTopographicMapsForAE.shape[2],1)

  data_normalized.extend(dataNormReshape)


data_normalized = np.array(data_normalized)

data_train, data_test = train_test_split(data_normalized, test_size=0.3, random_state=42, shuffle=True)
data_test, data_val = train_test_split(data_test, test_size=0.5, random_state=42, shuffle=True)

vae=autoencoder.fit(data_train, epochs=epoch, batch_size=batch_size, validation_data=(data_val, None), callbacks= [myCallBacks])

# extract encoder from autoencoder
encoder_trained = autoencoder.get_layer('encoder')
encoder_model_trained = Model(inputs=autoencoder.input, outputs=encoder_trained(encoder_input)[2])


predicted_wholedataset=autoencoder.predict(data_normalized).reshape(-1,data_normalized.shape[1],data_normalized.shape[2])
predicted_train=autoencoder.predict(data_train).reshape(-1,data_train.shape[1],data_train.shape[2])
#predicted_val=autoencoder.predict(data_val).reshape(-1,data_val.shape[1],data_val.shape[2])
predicted_test=autoencoder.predict(data_test).reshape(-1,data_test.shape[1],data_test.shape[2])

middle_vals_test = np.array(encoder_model_trained.predict(data_test))


# print("The BDM of the data_train: " + str(calculate_bdm(data_test, normalized=False)))
# print("The BDM of the predicted data: " + str(calculate_bdm(predicted_test, normalized=False)))

# print("The NBDM of the data_train: " + str(calculate_bdm(data_test, normalized=True)))
# print("The NBDM of the predicted data: " + str(calculate_bdm(predicted_test, normalized=True)))

actual_train=data_train.reshape(-1,data_train.shape[1],data_train.shape[2])
#actual_val=data_val.reshape(-1,data_val.shape[1],data_val.shape[2])
actual_test=data_test.reshape(-1,data_test.shape[1],data_test.shape[2])


#visualise actual and predicted image
fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(20,4))
for images, row in zip([actual_test[100:105], predicted_test[100:105]], axes):
  for img, ax in zip(images, row):
    ax.imshow(img.reshape((40, 40)))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('ActualPredicted5Images')


#Compute SSIM, MSE, and MAE
ssim_result = []
mse_result = []
mae_result = []
mape_result = []

SSIM_train=SSIM(actual_train,predicted_train)
print("SSIM completed")

MSE_train=MSE(actual_train,predicted_train)
print("MSE completed")

MAE_train=MAE(actual_train,predicted_train)
print("MAE completed")

MAPE_train=MAPE(actual_train,predicted_train)
print("MAPE completed")

SSIM_test=SSIM(actual_test,predicted_test)
print("SSIM completed")

MSE_test=MSE(actual_test,predicted_test)
print("MSE completed")

MAE_test=MAE(actual_test,predicted_test)
print("MAE completed")

MAPE_test=MAPE(actual_test,predicted_test)
print("MAPE completed")


header=  []
result = []


header.append('Dimension')
header.append('Training SSIM')
header.append('Training MSE')
header.append('Training MAE')
header.append('Training MAPE')
header.append('Testing SSIM')
header.append('Testing MSE')
header.append('Testing MAE')
header.append('Testing MAPE')


result.append(z_dim)
result.append(SSIM_train)
result.append(MSE_train)
result.append(MAE_train)
result.append(MAPE_train)
result.append(SSIM_test)
result.append(MSE_test)
result.append(MAE_test)
result.append(MAPE_test)
with open('test_all.csv', 'a') as f:
  writer = csv.writer(f)
  #header = ['Time_Slice_length','Dimension','SSIM', 'MSE','Training Accuracy', 'Training F1-Score', 'Training Precision', 'Training Recall','Validation Accuracy']
  #writer.writerow(header)
  #write the header
  writer.writerow(header)
  #write the data
  writer.writerow(result)


header=[]

header.append('Num')
header.append('Dimension')
header.append('Actual Data BDM')
header.append('Encoded Data BDM')
header.append('Predicted Data BDM')
header.append('Actual Data NBDM')
header.append('Encoded Data NBDM')
header.append('Predicted Data NBDM')

with open('test_all_test_bdms.csv', 'a') as f:
  writer = csv.writer(f)
  #header = ['Time_Slice_length','Dimension','SSIM', 'MSE','Training Accuracy', 'Training F1-Score', 'Training Precision', 'Training Recall','Validation Accuracy']
  #writer.writerow(header)
  #write the header
  writer.writerow(header)
  #write the data
  for i in range(actual_test.shape[0]):
    result=[]
    result.append(i)
    result.append(z_dim)
    result.append(calculate_bdm(actual_test[i]))
    result.append(calculate_bdm(middle_vals_test[i]))
    result.append(calculate_bdm(predicted_test[i]))
    result.append(calculate_bdm(actual_test[i], normalized=True))
    result.append(calculate_bdm(middle_vals_test[i], normalized=True))
    result.append(calculate_bdm(predicted_test[i], normalized=True))
    writer.writerow(result)
  writer.writerow([])

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
print("execution finished")
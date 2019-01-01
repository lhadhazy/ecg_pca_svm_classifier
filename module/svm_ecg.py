from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
# sys.path.append("..")
import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import peakutils
from sklearn.decomposition import PCA

def create_DS(ds_num, v_pre, v_post, feature_extract_methods=['RAW'], add_rri=True, cleanse=False):
    ds1_files = ['101','106','108','109','112','114','115','116','118','119','122','124','201','203','205','207','208','209','215','220','223','230','107','217']
    #ds1_files = ['101','106','108','109','112','114','115','116','118','119','122','124','201']
    ds2_files = ['100','103','105','111','113','117','121','123','200','202','210','212','213','214','219','221','222','228','231','232','233','234','102','104']
    #ds2_files = ['100','103','105','111','113','117','121','123','200','202','210','212','213']
    
    #initialize variables
    freq = 360
    preX = v_pre
    postX = v_post
    dfall = {} 
    dfann = {} 
    dfseg = {} 
    segment_data = []
    training_inputs1 = []
    training_inputs2 = []
    segment_labels = []
    if (ds_num == "1"):
        ds_list = ds1_files;
    else:
        ds_list = ds2_files;
    
    # Load the necessary patient inputs    
    for patient_num in ds_list:
        dfall[patient_num] = pd.read_csv('data/DS' + ds_num + '/' + patient_num + '_ALL_samples.csv', sep=',', header=0, squeeze=False)
        dfann[patient_num] = pd.read_csv('data/DS' + ds_num + '/' + patient_num + '_ALL_ANN.csv', sep=',', header=0, parse_dates=[0], squeeze=False)
   
    # Butterworth filter: x -> y
    lowcut=0.01
    highcut=15.0
    signal_freq=360
    filter_order=1
    nyquist_freq = 0.5*signal_freq
    low=lowcut/nyquist_freq
    high=highcut/nyquist_freq
    b, a = signal.butter(filter_order, [low,high], btype="band")
   
    # DWT specific section
    w = pywt.Wavelet('db4')

    # Standardize the beat annotations 
    # vals_to_replace = {'N':'N','L':'N','e':'N','j':'N','R':'N','A':'SVEB','a':'SVEB','J':'SVEB','S':'SVEB','V':'VEB','E':'VEB','F':'F','Q':'Q','P':'Q','f':'Q','U':'Q'}
    # use integers 0..4 instead of annotation...
    vals_to_replace = {'N':0, 'L':0, 'e':0, 'j':0, 'R':0, 'A':1, 'a':1, 'J':1, 'S':1, 'V':2, 'E':2, 'F':3, 'Q':4, 'P':4, 'f':4, 'U':4}
    
    for patient_num in ds_list:
        dfann[patient_num]['Type'] = dfann[patient_num]['Type'].map(vals_to_replace)    
        dfann[patient_num]['preRRI'] = (dfann[patient_num]['sample'] - dfann[patient_num]['sample'].shift(1)) / 360
        dfann[patient_num]['postRRI'] = (dfann[patient_num]['sample'].shift(-1) - dfann[patient_num]['sample']) / 360
        dfann[patient_num] = dfann[patient_num][1:]
        dfann[patient_num] = dfann[patient_num][:-1]    
    
    for patient_num in ds_list:
        annList = [];
        prerriList = [];
        postrriList = [];
        begNList = [];
        endNList = [];
        mixNList = [];
        sliceNList = [];

        for index, row in dfann[patient_num].iterrows():
            Nbegin = row['sample'] - preX;
            Nend = row['sample'] + postX;
            begNList.append(Nbegin);
            endNList.append(Nend);
            annList.append(row['Type'])
            prerriList.append(row['preRRI'])
            postrriList.append(row['postRRI'])       
                     
        #=======================================================================
        # for index, row in dfann[patient_num].iterrows():
        #     Nbegin = row['sample'] - preX;
        #     Nend = row['sample'] + postX;
        #     begNList.append(Nbegin);
        #     endNList.append(Nend);
        #     annList.append(row['Type'])
        #     rriList.append(row['RRI'])
        #=======================================================================

        mixNList = tuple(zip(begNList, endNList, annList, prerriList, postrriList))
  
        for row in mixNList:
            dfseg = dfall[patient_num][(dfall[patient_num]['sample'] >= row[0]) & (dfall[patient_num]['sample'] <= row[1])]
            dfseg1 = dfseg[dfseg.columns[1:2]]
            dfseg2 = dfseg[dfseg.columns[2:3]]
            if (cleanse == True):
                dfseg1_fir = signal.lfilter(b, a, dfseg[dfseg.columns[1:2]])
                dfseg2_fir = signal.lfilter(b, a, dfseg[dfseg.columns[2:3]])
                dfseg1_baseline_values = peakutils.baseline(dfseg1_fir)
                dfseg2_baseline_values = peakutils.baseline(dfseg2_fir)
                dfseg1 = dfseg1_fir-dfseg1_baseline_values
                dfseg2 = dfseg2_fir-dfseg2_baseline_values
            if 'RAW' in feature_extract_methods:
                if (cleanse == False):
                    training_inputs1 = np.asarray(dfseg1.values.flatten(), dtype=np.float32)
                    training_inputs2 = np.asarray(dfseg2.values.flatten(), dtype=np.float32)
                else:
                    training_inputs1 = np.asarray(dfseg1.flatten(), dtype=np.float32)
                    training_inputs2 = np.asarray(dfseg2.flatten(), dtype=np.float32)
            if 'DWT' in feature_extract_methods:
                if (cleanse == False):
                    training_inputs1 = np.asarray(dfseg1.values.flatten(), dtype=np.float32)
                    training_inputs2 = np.asarray(dfseg2.values.flatten(), dtype=np.float32)
                else:
                    training_inputs1 = np.asarray(dfseg1.flatten(), dtype=np.float32)
                    training_inputs2 = np.asarray(dfseg2.flatten(), dtype=np.float32)
                    
                training_inputs1_coeffs = pywt.wavedec(training_inputs1, w, level=5 )
                training_inputs1 = np.concatenate([training_inputs1_coeffs[0],training_inputs1_coeffs[1]])
                training_inputs2_coeffs = pywt.wavedec(training_inputs2, w, level=5 )
                training_inputs2 = np.concatenate([training_inputs2_coeffs[0],training_inputs2_coeffs[1]])
            if (add_rri == True):
                training_inputs2 = np.concatenate((training_inputs2, np.asarray([row[3]], dtype=np.float32)))
                training_inputs2 = np.concatenate((training_inputs2, np.asarray([row[4]], dtype=np.float32)))
            segment_data.append(np.concatenate((training_inputs1, training_inputs2), axis=0))
            training_labels = row[2]
            segment_labels.append(training_labels)    
            
    segment_data = np.asarray(segment_data)
    
    if 'PCA' in feature_extract_methods:   
        pca = PCA(n_components=16)
        segment_data = pca.fit_transform(segment_data)
            
    return dfall, dfann, segment_data, segment_labels


def main():

    preX = 89
    postX = 150
    ds1_all, ds1_ann, ds1_seg, ds1_lab = create_DS("1",preX,postX,feature_extract_methods=['RAW', 'PCA'],add_rri=True, cleanse=False)
    ds2_all, ds2_ann, ds2_seg, ds2_lab = create_DS("2",preX,postX,feature_extract_methods=['RAW', 'PCA'],add_rri=True, cleanse=False)




if __name__ == "__main__":
    main()
    
    

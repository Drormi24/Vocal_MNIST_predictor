import os
from utils.WAV2IMG import display_WAV,WAV2FFT,get_Spectrogram
import numpy as np
from PIL import Image

def split_data(X,Y,p=0.7):
    perm = np.random.permutation(len(Y))
    X = X[perm,:]
    Y = Y[perm]
    split = int(p * len(Y))
    train = (X[:split,:] , Y[:split])
    test = (X[split:,:] , Y[split:])
    return train,test

def WAV_DIR_2_IMG_DIR(audio_path,img_path,img_size,p,show_flag,save_flag):
    # initialize labels vector, 'Y' and features array, 'img_array'
    Y = []
    img_array = np.empty((0,img_size*img_size))
    
    # main loop: go over path directory and peform the following:
    for i,filename in enumerate(os.listdir(audio_path)):
        f = os.path.join(audio_path,filename)
        # checking if it is a file
        if os.path.isfile(f): # verify its a file
            
            # display audio signal and/or display its FFT conversion
            # display_WAV(f)
            # WAV2FFT(f)
            if save_flag == 'Yes':
                # generate a spectrogram and save it an image file     
                img = get_Spectrogram(f,show_flag)
                Image.Image.save(img,(img_path+'audio_images'+str(i)+'.jpg'))
            
            # get its label from its name and seperate it from features
            f_name = filename[0]
            Y = np.append(Y,f_name)
            
            # prepare images for CNN learning process     
            img = Image.open((img_path+'audio_images'+str(i)+'.jpg'))
            img = img.resize((img_size,img_size),Image.ANTIALIAS)
            img = img.convert('L')
            img = np.array(img).reshape(1,-1).astype(int)
            img_array = np.append(img_array,np.array(img),axis=0).astype(int)
            
        else:
            break
        
    # save all features into a CSV file    
    np.savetxt('img_csv_data.csv',img_array,delimiter=',')
    
    # randomly split data as train and test acc. p 
    train,test = split_data(img_array,Y,p)
    return train,test  # return 2 tuples: train / test: (features,labels)


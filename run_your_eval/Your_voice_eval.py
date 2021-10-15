from utils.WAV2IMG import display_WAV,WAV2FFT,get_Spectrogram
from tensorflow.keras import models
from PIL import Image
import os
import numpy as np
import warnings

warnings.filterwarnings('always')

# define paths to audio files and spctrogram images files
audio_path = 'run_your_eval/recordings'
img_path = 'run_your_eval/images/'
model_path = 'utils/my_model'
show_flag = 'True'  # show or not spctrogram results
img_size = 28       # image size must be 28*28*1
num_classes = 10    #number of classes
Y = []
img_array = np.empty((0,img_size*img_size))

# main loop: go over your recordings and convert them to FFT-based images 
# and to numeric arrays
for count,filename in enumerate(os.listdir(audio_path)):
    f = os.path.join(audio_path,filename)
    # checking if it is a file
    if os.path.isfile(f): # verify its a file
        
        # display audio signal and/or display its FFT conversion
        display_WAV(f)
        WAV2FFT(f)
        
        # generate a spectrogram and save it an image file     
        img = get_Spectrogram(f,show_flag)
        Image.Image.save(img,(img_path+str(count)+'.jpg'))
        
        # get its label from its name and seperate it from features
        f_name = filename[0]
        Y = np.append(Y,f_name).astype(int)
        
        img = Image.open((img_path+str(count)+'.jpg'))
        img = img.resize((img_size,img_size),Image.ANTIALIAS)
        img = img.convert('L')
        img = np.array(img).reshape(1,-1).astype(int)
        img_array = np.append(img_array,np.array(img),axis=0).astype(int)
        
    else:
        break
    
# prepare numeric array to CNN model evaluation    
X = (img_array.astype('float32')/255).reshape(-1,img_size,img_size)
X = np.expand_dims(X,-1)

# upload CNN model including its weights
recon_model = models.load_model(model_path) 

# predict   
Y_predict = recon_model.predict_classes(X)
score = np.zeros((1,len(Y)))
score = [1 for a,b in zip(Y_predict,Y) if a==b]
accuracy = sum(score)/len(Y)*100

# show results
print(f'Predicted values are: {Y_predict}...while True values are: {Y}')
print(f'Your voice MNIST test accuracy is: {round(accuracy,2)}%')





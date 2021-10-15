from utils.WAV_DIR_2_IMG_DIR import WAV_DIR_2_IMG_DIR
from utils.CNN_model import get_CNN_model
from tensorflow.keras import models

# global params
# define paths to audio files and spctrogram images files
audio_path = 'C:/Users/frido/Documents/GitHub/FSDD/data/'
img_path = 'C:/Users/frido/Documents/GitHub/FSDD/audio_images/'
model_path = 'C:/Users/frido/Documents/GitHub/FSDD/utils/my_model'
p = 0.7           # train data split % factor 
num_classes = 10  # number of classes to be predicted
img_size = 28     # image size. NOTE! if changed-model structure will need adjustments
batch_size = 128  # CNN run param: how many images will be batch per run
epochs = 500      # number of learning cycles
valid_split = 0.1 # % of validated data per batch for expedited learning
show_flag = 'No'  # show or not spctrogram results [Yes/No]
save_flag = 'No'  # create and save images? [Yes/No]

# convert audio signal to spctrogram image and prepare data for a CNN run
train,test = WAV_DIR_2_IMG_DIR(audio_path,img_path,img_size,p,show_flag,save_flag)

# create CNN model
model,x_train,y_train,x_test,y_test = get_CNN_model(train,test,num_classes,img_size)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=valid_split)
    
score = model.evaluate(x_test,y_test,verbose=0)
print(f'Test loss:{score[0]}')
print(f'Test accuracy: {round(score[1],2)}%')
models.save_model(model,model_path)


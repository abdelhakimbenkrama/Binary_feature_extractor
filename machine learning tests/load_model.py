#%%
from keras.models import model_from_json
# load json and create model
#You need to train the model first
path = 'machine learning tests/model.json'
json_file = open(path, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("machine learning tests/model.h5")

#%%
from PIL import  ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True
# load data using imagedatagenerator
link = "C:\\Users\\NEW.PC\Desktop\datasets\\2D_images_dataset_FE"
batch_size = 10
nb_epochs = 1

train_gen = ImageDataGenerator(rescale=1. / 255,
                               rotation_range=40,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               validation_split=0.2)

train_generator = train_gen.flow_from_directory(
    directory=link,
    class_mode='categorical',
    target_size=(224, 224),
    subset='training',
    batch_size=batch_size
)
valid_generator = train_gen.flow_from_directory(
    directory=link,
    class_mode='categorical',
    target_size=(224, 224),
    subset='validation',
    batch_size=batch_size
)
#%%
loaded_model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["acc"])

loaded_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=nb_epochs)

#%%
model_json = loaded_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
loaded_model.save_weights("model.h5")
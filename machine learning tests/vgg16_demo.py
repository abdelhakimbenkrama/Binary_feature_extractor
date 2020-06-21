#%%
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.regularizers import l2,l1
from PIL import  ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#%%
def getModel(output_dim):
    # output_dim: the number of classes (int)
    # return: compiled model (keras.engine.training.Model)

    vgg_model = VGG16(weights='imagenet', include_top=True )
    vgg_out = vgg_model.layers[-1].output
    #     vgg_out = Flatten()(vgg_out)

    vgg_out = Dropout(0.25)(vgg_out)
    # Create new transfer learning model
    out = Dense(output_dim, activation="softmax", W_regularizer=l2(0.2))(vgg_out)

    tl_model = Model(input=vgg_model.input, output=out)
    for layer in tl_model.layers[0:-1]:
        layer.trainable = False

    tl_model.compile(loss="categorical_crossentropy", optimizer="adagrad", metrics=["acc"])
    print (tl_model.summary())
    return tl_model
model = getModel(102)
#%%


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
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    epochs=nb_epochs)

#%%
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
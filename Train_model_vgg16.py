import matplotlib.pyplot as plt
import os
from keras.applications import vgg16
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import regularizers
from keras.optimizers import Adam

food_labels = ["char_koay_teow",
               "deep_fried_shrimp",
               "four_joy_meatball",
               "fried_dumpling",
               "hamburger",
               "laver_wrapped_rice_roll",
               "mango_glutinous_rice",
               "nasi_lemak",
               "popiah",
               "roasted_chicken_wings",
               "roasted_leek",
               "roti_canai",
               "satay",
               "spaghetti_bolognese",
               "steamed_egg_custard",
               "steamed_fish",
               "stir_fry_lotus_root",
               "toast_bread"]

base_dir = r"C:\Users\sptio\Documents\SEM 8 (Sep 2022)\TSE30910 SE PROJECT\data_split_8_2"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=vgg16.preprocess_input
        )

validate_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input)

batch_size = 32

# flow from directory obtains images from the specified folders in the specified batch size.
# flow_from_directory provides to labels image, based on the name of the folder in which the image is present.
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    class_mode='categorical',
                                                    classes=food_labels,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=42)

validate_generator = validate_datagen.flow_from_directory(val_dir,
                                                          target_size=(224, 224),
                                                          class_mode='categorical',
                                                          classes=food_labels,
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          seed=42)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  class_mode=None,
                                                  classes=food_labels,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  seed=42)

# the VGG16 is loaded as base model and neural network layers are added
base_model = vgg16.VGG16(weights="imagenet",
                         include_top=False,
                         input_shape=(224, 224, 3))
base_model.trainable = False

# inputs = keras.layers.Input(shape=(224, 224, 3))
head_model = base_model.output
head_model = layers.Flatten()(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(256, kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3), activation='relu')(head_model)
head_model = layers.Dropout(0.5)(head_model)
head_model = layers.Dense(18, activation='softmax', name="output")(head_model)    # classes in dataset
head_model = head_model

base_model.summary()

model = Model(inputs=base_model.input, outputs=head_model)
base_model.trainable = True

set_trainable = False
for layer in base_model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True  # after black5_conv1, set_trainable becomes True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# the loss, optimzer and metric for evaluation are set
model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=3e-5),   # Low learning rate
              metrics="accuracy")

vgg_history = model.fit(
    train_generator,
    batch_size=batch_size,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=100,
    validation_data=validate_generator,
    validation_steps=validate_generator.n // validate_generator.batch_size,
    verbose=1
)

acc = vgg_history.history['accuracy']
val_acc = vgg_history.history['val_accuracy']
loss = vgg_history.history['loss']
val_loss = vgg_history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()         # Close figure for test to continue run
print('The figure is shown')

model.save('vgg16_food_model.h5')
print('Model is saved.')

model.summary()
test_loss, test_accuracy = model.evaluate(validate_generator)

print("Test Loss of Model:", test_loss)
print("Test Accuracy of Model:", test_accuracy)




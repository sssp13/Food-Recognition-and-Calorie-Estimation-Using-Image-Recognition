import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluation of vgg16 model

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

# Load model
vgg_16_model = load_model("vgg16_food_model.h5")

# In generator the "shuffle" parameter when evaluating a model,
# has to be set "False", so that the predictions from model and true_labels obtained are for the same image.

# If the "shuffle" = True, the order of images predicted from data sub set (train/test) changes.
train_generator_shuffle = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(224, 224),
                                                            class_mode='categorical',
                                                            classes= food_labels,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            seed=42)

validate_generator_shuffle = validate_datagen.flow_from_directory(val_dir,
                                                                  target_size=(224, 224),
                                                                  class_mode='categorical',
                                                                  classes=food_labels,
                                                                  batch_size=batch_size,
                                                                  shuffle=False,
                                                                  seed=42)

test_generator_shuffle = test_datagen.flow_from_directory(test_dir,
                                                          target_size=(224, 224),
                                                          class_mode=None,
                                                          classes=food_labels,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          seed=42)


# a function to evaluate the prediction of dataset from a given model
def model_evaluation(model, generator):
    generator.reset()
    # There is a reset() method for the datagenerators which resets it to the first batch.
    # So whenever you would want to correlate the model output with the filenames you need to set shuffle as False and
    # reset the datagenerator before performing any prediction. This will ensure that our files are being read properly
    # and there is nothing wrong with them.

    true_classes = generator.classes
    class_indices = dict((v, k) for k, v in generator.class_indices.items())
    preds = model.predict(generator)
    preds_classes = np.argmax(preds, axis=1)
    return [true_classes, preds_classes, generator.class_indices.keys()]


# for the prediction made, this function prints a confusion matrix
def plot_confusion_matrix(true_classes, preds_classes, target_names):
    cm = confusion_matrix(true_classes, preds_classes)

    df_cm = pd.DataFrame(cm, columns=target_names, index=target_names)
    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1.4) # for label size
    sns.heatmap(df_cm, annot=True, fmt='g', cbar=False, cmap="YlGnBu") # font size , annot_kws={"size": 16}
    plt.title('Confusion Matrix\n', y=1.1)
    plt.ylabel('Actual label\n')
    plt.xlabel('Predicted label\n')

    plt.show()


result_1 = model_evaluation(vgg_16_model, train_generator_shuffle)
true_classes_1, preds_classes_1, target_names_1 = result_1[0], result_1[1], result_1[2]
train_acc = accuracy_score(true_classes_1, preds_classes_1)
print("VGG-16 Model without Fine-Tuning on Training set has Accuracy: {:.2f}%".format(train_acc * 100))

result_2 = model_evaluation(vgg_16_model, test_generator_shuffle)
true_classes_2, preds_classes_2, target_names_2 = result_2[0], result_2[1], result_2[2]
validate_acc = accuracy_score(true_classes_2, preds_classes_2)
print("VGG-16 Model without Fine-Tuning on Testing set has Accuracy: {:.2f}%".format(validate_acc * 100))


print("Classification report of Testing set without Fine-Tuning : ")
print(classification_report(true_classes_2, preds_classes_2, target_names=target_names_2))


print("Confusion Matrix of Testing set without Fine-Tuning : ")
plot_confusion_matrix(true_classes_2, preds_classes_2, target_names_2)

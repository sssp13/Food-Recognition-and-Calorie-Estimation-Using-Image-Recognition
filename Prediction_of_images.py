import numpy as np
from keras.models import load_model
import calories
from keras.applications import vgg16
from keras.utils import load_img, img_to_array

# Predict image
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
# Load model
vgg_16_model = load_model("vgg16_food_model.h5")


def predict_image(path):
    img = load_img(path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = vgg16.preprocess_input(img_batch)
    prediction = vgg_16_model.predict(img_preprocessed)
    predict_food = food_labels[np.argmax(prediction)]
    return predict_food


def calories_estimation(label):
    img = predict_image(label)
    if img == 'char_koay_teow':
        food_name = calories.char_koay_teow[0]
        calorie = calories.char_koay_teow[1]
        return food_name, calorie

    if img == 'fried_dumpling':
        food_name = calories.fried_dumpling[0]
        calorie = calories.fried_dumpling[1]
        return food_name, calorie

    if img == 'hamburger':
        food_name = calories.hamburger[0]
        calorie = calories.hamburger[1]
        return food_name, calorie

    if img == 'laver_wrapped_rice_roll':
        food_name = calories.laver_wrapped_rice_roll[0]
        calorie = calories.laver_wrapped_rice_roll[1]
        return food_name, calorie

    if img == 'nasi_lemak':
        food_name = calories.nasi_lemak[0]
        calorie = calories.nasi_lemak[1]
        return food_name, calorie

    if img == 'roti_canai':
        food_name = calories.roti_canai[0]
        calorie = calories.roti_canai[1]
        return food_name, calorie

    if img == 'satay':
        food_name = calories.satay[0]
        calorie = calories.satay[1]
        return food_name, calorie

    if img == 'spaghetti_bolognese':
        food_name = calories.spaghetti_bolognese[0]
        calorie = calories.spaghetti_bolognese[1]
        return food_name, calorie

    if img == 'steamed_fish':
        food_name = calories.steamed_fish[0]
        calorie = calories.steamed_fish[1]
        return food_name, calorie

    if img == 'toast_bread':
        food_name = calories.toast_bread[0]
        calorie = calories.toast_bread[1]
        return food_name, calorie

    else:
        food_name = "food not found"
        calorie = 0
        return food_name, calorie


def prediction_food_calories(file):
    food_name, food_calorie = calories_estimation(file)
    print("Prediction of food:", food_name)
    if food_name == "food not found":
        return "Not Available"

    else:
        a = []
        for i in range(5):
            i = i + 1
            gram = i*100
            calorie = food_calorie*i
            b = '\n {} g = {} kcal '.format(gram, calorie)
            a.append(b)
            # a.append(b)
        f_calorie = a
        print("Calories:", a[0], a[1], a[2], a[3], a[4])
        return ('{} {} {} {} {} {}'. format(food_name, a[0], a[1], a[2], a[3], a[4]))

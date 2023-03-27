#!/usr/bin/python3
# coding: utf-8

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import PIL.Image
import pathlib


class myScript(object):
    """
    Constructeur de classe
    """

    def __init__(self):
        """Definis les attributs!"""
        # initialize mains objects here
        self.parameters = None

    def main(self):
        """
        Le main de ce script realise, à partir d'une image,	la prediction des probabilités d'appartenance à une des
         13 race de chien du dataset stanford dogs en utilisant un reseaux de convolution DenseNet.
        Se réseaux s'appuie sur du transfert learning.
        """
        self.parseOptions()
        #
        # Load your model
        DenseNetModel = tf.keras.models.load_model('DenseNet_20epoch_93acc')
        #
        if self.parameters.Evaluate == 0:
            # chargement de l'image
            in_file = PIL.Image.open(self.parameters.input, mode='r')
            # Loading train class as a name
            # Preprocessing removed them
            open_file = open("trainclass.pkl", "rb")
            train_class = pickle.load(open_file)
            open_file.close()
            #
            # predict
            preprocessed_images = self.preprocessing(in_file)
            probabilities = DenseNetModel.predict(preprocessed_images)
            #
            df_results = self.formating_results(train_class, probabilities)
            print(df_results)
            #
            if self.parameters.output != 0:
                df_results.to_csv(self.parameters.output)

        else:
            eval_data = tf.keras.utils.image_dataset_from_directory(pathlib.Path("EvalMode/"))
            eval_data_std = eval_data.map(self.preprocess_densenet_for_eval)
            loss, acc = DenseNetModel.evaluate(eval_data_std)
            print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    def preprocessing(self, img_test):
        """
        Fonction for preprocessing image in input to DenseNet pretrained model
        :param img_test: image input
        :return: preprocessed image ready to predict in DenseNetModel
        """
        resizing = np.array(img_test.convert("RGB").resize((256, 256)))
        image_with_batch_to_tensor = np.expand_dims(resizing, axis=0)
        preprocessed = tf.keras.applications.densenet.preprocess_input(image_with_batch_to_tensor)
        #
        return preprocessed

    def formating_results(self, dog_class, proba_to_print):
        """
        fonction pour preparer les résultats et les renvoyers sous forme de DataFrame pandas

        :param dog_class: class from train
        :param proba_to_print: probabilities from predict
        :return: pandas dataframe
        """
        table_for_save = {}
        #
        for index, races in enumerate(dog_class):
            table_for_save.update({races.split("-")[1]: f"{proba_to_print[0][index]:.3f}"})
        #
        df_results = pd.DataFrame.from_dict(table_for_save, orient='index')
        return df_results

    def preprocess_densenet_for_eval(self, images, labels):
        """
        This function is only for eval
        """
        return tf.keras.applications.densenet.preprocess_input(images), labels

    def parseOptions(self):
        """
        parse les options du script avec le module argparse
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", help="Input file", required=True)
        parser.add_argument("-o", "--output", help="name of output file", default=0)
        parser.add_argument("-E", "--Evaluate", help="DevTools for evaluation, input requirement need to be lifted", default=0)
        self.parameters = parser.parse_args()


if __name__ == "__main__":
    """
    This code block will only be executed if template.py is called as a script
    (ie. it will not be executed if the template class is just imported by another module)
    """
    # build the script object
    myScriptRunner = myScript()
    # run execute the main method
    myScriptRunner.main()

#importation des librairies
from keras.models import Sequential		#--initalisation de mon model NN
from keras.layers import Conv2D			#--convolution sur les images
from keras.layers import MaxPooling2D	#--operation de pooling (aplatination)
from keras.layers import Flatten		#--conversion de tableau2D en vecteur lineaire long continu
from keras.layers import Dense			#--conection entiere pour build le CNN
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#__1er arg = nombre de filtres [32]
#__2e arg = forme de chauq efiltres [3x3]
#__3e arg = type d'entree et le type d'image de chq img [resolution=64x64] [3 correspd a RGB]
#__4e arg fonction d'activation [relu]
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#__ajout de couche de regroupement
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu')) 	#__couche cahcée de 128 noeuds 
classifier.add(Dense(units = 1, activation = 'sigmoid')) #__chat ou chien [couche de sortie comportant 1noeud]
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
								   shear_range = 0.2,
								   zoom_range = 0.2,
								   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('training_set',
												target_size = (64, 64),
												batch_size = 32,
												class_mode = 'binary')
test_set = test_datagen.flow_from_directory('test_set',
												target_size = (64, 64),
												batch_size = 32,
												class_mode = 'binary')

classifier.fit_generator(training_set,
						 steps_per_epoch = 8000,
						 epochs = 25,
						 validation_data = test_set,
						 validation_steps = 2000)

#__steps_per_epoch = [nombre d'image ds training_set]
#__epoch = [25]formation d1 NN

import numpy as np
from keras.preprocessing import image
test_image = image.load('single_prediction/cat_or_dog_1.jpg',
						target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
	prediction = 'dog'
	print("__--__ "+prediction+" __--__")
else:
	prediction = 'cat'
	print("__--__ "+prediction+" __--__")

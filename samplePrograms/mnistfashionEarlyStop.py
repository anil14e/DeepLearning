import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True


mnist = tf.keras.datasets.fashion_mnist

(training_images,training_labels),(test_images,test_labels) = mnist.load_data()


import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt

training_images = training_images/1.0
test_images = test_images/1.0

## Normalize input data
training_images = training_images/255.0
test_images = test_images/255.0


callbacks = myCallback()

##Define Model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(512, activation=tf.nn.relu),tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(training_images,training_labels, epochs=10,callbacks=[callbacks])

print(model.evaluate(test_images,test_labels))


classifications = model.predict(test_images)

print('classifications')
print(classifications[0])


print('Test')
print(test_labels[0])

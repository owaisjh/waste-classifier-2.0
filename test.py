import tensorflow as tf

MODEL_FILENAME = "model.hdf5"

model = tf.keras.models.load_model(MODEL_FILENAME)
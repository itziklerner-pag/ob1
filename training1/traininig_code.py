import pandas as pd
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.mixed_precision import set_global_policy

# Configuration
config = {
    "data_path": "/content/drive/MyDrive/tv_colab/XY30000.csv",
    "output_dir": "/content/drive/MyDrive/tv_colab/output",
    "checkpoint_path": "checkpoint.pth",
    "best_model_path": "best_model.pth",
    "batch_size": 128,
    "time_steps": 600,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "num_features": 80,
    "num_classes": 3,
    "shuffle": True,
    "save_interval": 1,  # Save checkpoint every 2 epochs
    "log_interval": 100,  # Log metrics every 100 batches
    "policy": 'mixed_float16'
}

# Enable mixed precision

set_global_policy(config["policy"])



mydrivepath = '/content/drive/MyDrive/tv_colab'
# Paths for saving models
checkpoint_filepath = './checkpoint'
final_model_path = './final_model.keras'

# Load your data
data = pd.read_csv('./XY30000.csv', header=None)

# Assuming the last column is the label and the rest are features
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Generator function to create sequences
def sequence_generator(X, y, time_steps=config["time_steps"], batch_size=config["batch_size"]):
    num_batches = (len(X) - time_steps) // batch_size
    while True:  # Infinite loop for keras model training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size + time_steps
            X_batch = np.array([X[i:(i + time_steps)] for i in range(start_idx, end_idx - time_steps)])
            y_batch = np.array([y[i + time_steps] for i in range(start_idx, end_idx - time_steps)])
            yield X_batch, y_batch

# Define the Transformer part
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(tf.cast(x, tf.float16) + tf.cast(inputs, tf.float16))
    #x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)


    # Feed Forward Part
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dropout(dropout)(ff)
    ff = layers.Dense(inputs.shape[-1])(ff)
    #ff = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    ff = layers.LayerNormalization(epsilon=1e-6)(tf.cast(x, tf.float16) + tf.cast(ff, tf.float16))
    return ff

# Build the Model
def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)

# Initialize the generator
train_gen = sequence_generator(X_scaled, y_encoded, time_steps=config["time_steps"], batch_size=config["batch_size"])

# Model configuration
input_shape = (config["time_steps"], X_scaled.shape[1])  # Input shape should match the sequence length and feature count
head_size = 256
num_heads = 4
ff_dim = 256
num_transformer_blocks = 4
mlp_units = [128]
dropout = 0.1
mlp_dropout = 0.1

model = build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout)
#optimizer = tf.keras.optimizers.Adam()
optimizer = tf.keras.optimizers.legacy.Adam()
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ModelCheckpoint to save the model during training
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1)

# EarlyStopping to stop training when the validation accuracy does not improve
early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    verbose=1,
    mode='max',
    restore_best_weights=True)

# Training the model
steps_per_epoch = (len(X_scaled) - config["time_steps"]) // config["batch_size"]
# Assuming your classes are labeled as 1, 2, 3 in the dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights_dict = {i : class_weights[i] for i in range(len(class_weights))}

# Training the model with the callbacks
history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=config["num_epochs"],
    validation_data=train_gen,
    validation_steps=50,
    class_weight=class_weights_dict,
    callbacks=[model_checkpoint_callback, early_stopping_callback])

model.save(final_model_path)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()



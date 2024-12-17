import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Input
from keras.layers import LSTM, Dense, Attention, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.utils import Sequence
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

"""
2 Different models are tried for this dataset.
Dataset has 3 axes
X axis is Timesteps. Each timestep is a 5 minute interval. Each data has 2880 time steps
Y axis are Different channels. Each channel has a normalized exchange data (with various methods like calculating z-score) for the corresponding time step.
These normalized datas are between -1 and 1
Z axis is the number of data

So the data we've used here is 2D sections of crypto exchange data. One dimension is time axis and the other one has the custom indicators we created.
"""
# Lstm model gave better results for this dataset
class LSTMModel:
    def __init__(self,input_shape_x,input_shape_y,output_size, lstm_units, learning_rate,dropout_katsayi):
        self.input_shape_x = input_shape_x
        self.input_shape_y = input_shape_y
        self.output_size = output_size
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.dropout_katsayi = dropout_katsayi
        self.model = self.build_model()
        
    def build_model(self):
        tf.keras.backend.clear_session()
        initializer = tf.keras.initializers.GlorotUniform(seed=6)
        
        # Encoder model
        # 1 Layer of lstm
        encoder_inputs = Input(shape=(self.input_shape_x-1, self.input_shape_y), name='encoder_inputs') # Teacher enforcimg
        encoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True, name='encoder_lstm',
                            kernel_initializer=initializer,dropout = self.dropout_katsayi)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        states = [state_h, state_c] # these states of encoder layer will be passed to decoder input
        
        # Decoder model
        # 2 layers of lstm
        decoder_inputs = Input(shape=(self.input_shape_x-1, self.input_shape_y), name='decoder_inputs')
        decoder_lstm = LSTM(self.lstm_units, return_sequences=True, return_state=True, name='decoder_lstm',
                            kernel_initializer=initializer,dropout = self.dropout_katsayi)#,recurrent_initializer=rec_in)
        decoder_lstm2 = LSTM(self.lstm_units, return_sequences=True, name='decoder_lstm2',kernel_initializer=initializer,dropout =self.dropout_katsayi)
        # 1 dense layer at the end of the decoder layer
        decoder_dense = Dense(self.output_size, activation='softmax', name='outputs',kernel_initializer=initializer)
        outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=states)
        decoder_outputs =  decoder_lstm2(outputs)
        # decoder_outputs = BatchNormalization()(decoder_outputs)
        
        # Attention layer
        attention = Attention()
        attention_output = attention([encoder_outputs, decoder_outputs])
        
        # Final output
        decoder_outputs = decoder_dense(Flatten()(attention_output))        
        model_lstm = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        
        # categorical crossentropy and adam optimizer gave the best results for this dataset
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)        
        model_lstm.compile(loss = "categorical_crossentropy", optimizer=opt,metrics = ["accuracy"]) #accuracy
        model_lstm.summary()
        return model_lstm

# Since data kinda resembles an image (2D sections of crypto exchange graph), CNN can work for this dataset    
class CNNModel:
    def __init__(self, input_shape_x, input_shape_y, output_size, learning_rate, dropout_katsayi):
        self.input_shape_x = input_shape_x
        self.input_shape_y = input_shape_y
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_katsayi = dropout_katsayi
        self.model = self.build_model()

    def build_model(self):
        tf.keras.backend.clear_session()
        initializer = tf.keras.initializers.Orthogonal(gain=0.3, seed=61)

        # CNN layers
        # Small amount of big filters (large kernel size) gives the best result for this dataset in Conv2d
        inputs = Input(shape=(self.input_shape_x, self.input_shape_y, 1), name="input_layer")

        # Dataset has values between -1 and 1, so using leaky_relu or tanh as activation function yields better reuslts
        cnn1 = Conv2D(filters=15, kernel_size=(303, 3), activation="leaky_relu",
                             kernel_initializer=initializer)(inputs)
        cnn1 = MaxPooling2D(pool_size=(2, 1))(cnn1)

        cnn2 = Conv2D(filters=15, kernel_size=(101, 3), activation="leaky_relu",
                             kernel_initializer=initializer)(cnn1)

        cnn3 = Conv2D(filters=15, kernel_size=(37, 1), activation="leaky_relu",
                             kernel_initializer=initializer)(cnn2)

        # Flatten and Dense layers
        cnn_flatten = Flatten()(cnn3)
        outputs = Dense(self.output_size, kernel_initializer=initializer,
                               activation="softmax", name="output_layer")(cnn_flatten)

        # Model
        model_cnn = Model(inputs=inputs, outputs=outputs, name="cnn_model")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model_cnn.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        model_cnn.summary()

        return model_cnn


# Custom data generator to feed the data in batches, in order to save gpu memory
class DataGenerator(Sequence):
    # There are 2 inputs, 1 outğput
    def __init__(self, x_set1, x_set2, y_set, batch_size):
        self.x1,self.x2, self.y = x_set1, x_set2, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x1) / float(self.batch_size)))

    # returns next iter of inputs and outputs
    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [batch_x1,batch_x2], batch_y


def plot_confusion_matrix(real_data, predictions):
    labels=['long', 'short', 'nopos']
    cm = confusion_matrix(real_data, predictions)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=(7, 7))
    sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g', cbar=False)
    plt.title("LSTM Validation Confusion Matrix")
    plt.xlabel("Predicted Shape")
    plt.ylabel("True Shape")
    plt.show()    
    
def main():
    output_size = 2
    learning_rate = 0.001 # Default value is 0.01    
    
    dropout_katsayi = 0.5
    
    # Regularization did not have much impact on this 
    # default 0.01.     1 --> full regularization,      0 --> no regularization
    l2_reg = 0.01
    l1_reg = 0.01
        
    batch_size = 512
    lstm_units = 8
    
    # Load Data
    egitim_dataset = np.load('aaa') 
    egitim_etiket = np.load('aaa', allow_pickle = True) 
    
    dogrulama_dataset = np.load('bbb') 
    dogrulama_etiket = np.load('bbb', allow_pickle = True) 
    
    boyut_x = egitim_dataset.shape[0]   # X axis is Timesteps. Each timestep is a 5 minute interval. Each data has 2880 time steps
    boyut_y = egitim_dataset.shape[1]   # Different channels. Each channel has a  normalized exchange data (with various methods like calculating z-score) for the corresponding time step
    boyut_z = egitim_dataset.shape[2]   # number of data, batch size
    
    # Rearrange the input shape
    input_egitim = np.transpose(egitim_dataset[:,:,:], (2,0,1))
    input_dogrulama = np.transpose(dogrulama_dataset[:,:,:], (2,0,1))
    
    # Create data generator objects
    train_gen = DataGenerator(input_egitim[:, :-1, :],input_egitim[:, 1:, :], egitim_etiket, batch_size)
    val_gen = DataGenerator(input_dogrulama[:, :-1, :],input_dogrulama[:, 1:, :], dogrulama_etiket, batch_size)
    
    # Create the model
    model_lstm = LSTMModel(boyut_x,boyut_y,output_size, lstm_units, learning_rate, dropout_katsayi)    
    
    # Early stop using valdation accuracy
    early_stop = EarlyStopping(monitor='val_accuracy', patience=100, mode="max",restore_best_weights=True)
    
    model_lstm.fit(train_gen, batch_size = batch_size, epochs = 2500, verbose = 1, validation_data=val_gen,
                                callbacks=[CSVLogger('training_lstm.csv'),early_stop])

    # Plot confusion matrix
    test_predictions = model_lstm.predict([input_dogrulama[:, :-1, :],input_dogrulama[:, 1:, :]])
    predictions = test_predictions.argmax(axis=1) # predictions are one hot encoded, we convert them to integers 0,1 and 2
    real_data_test = dogrulama_etiket.argmax(axis=1) # same with the real data labels
    plot_confusion_matrix(real_data_test, predictions)


if __name__ == "__main__":
    main()

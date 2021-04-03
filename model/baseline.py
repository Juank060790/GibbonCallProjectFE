'''
Constructs a baseline model. 
'''
import tensorflow as tf 
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer 
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D, Input

def baselineModel(input_shape: tuple, conv_layers: int, conv_filters: int, 
          conv_kernels: int, dropout_rate: float, max_pooling_size: int, 
          fc_layers: int, fc_units: int, activation: str = "relu"):
    
    model = Sequential()
    model.add(Input(shape = input_shape))

    for i in range(conv_layers):
        model.add(
            Conv2D(filters = conv_filters, kernel_size = conv_kernels, 
            activation = activation)
        )
        model.add(Dropout(rate = dropout_rate))
        model.add(MaxPool2D(pool_size = max_pooling_size))       
    
    model.add(Flatten())
    for i in range(fc_layers):
        model.add(Dense(units = fc_units, activation = activation))
        model.add(Dropout(rate = dropout_rate))
    
    model.add(Dense(2, activation = "softmax"))
    model.compile(loss = "categorical_crossentropy", optimizer = "adam",
                  metrics = ["accuracy"])
    
    return model 

def network():
    input_shape = (118, 188, 1)
    conv_layers = 2
    fc_layers = 1
    max_pooling_size = 4
    dropout_rate = 0.4
    conv_filters = 8
    conv_kernels = 16
    fc_units = 32

    model = baselineModel(input_shape, conv_layers, conv_filters, conv_kernels,
                  dropout_rate, max_pooling_size, fc_layers, fc_units)

    return model 

if __name__ == "__main__":
    model = network()  
        

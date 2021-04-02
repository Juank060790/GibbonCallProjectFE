import tensorflow as tf 
from tensorflow.keras import Model 
from tensorflow.keras.layers import Layer 
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPool2D, Conv2D 

class BaseLine(Model):
    def __init__(self, name: str = "Baseline", input_shape = (128, 188, 1),
                 conv_layers: int, conv_filters: int, conv_kernel: int, 
                 fc_layers: int, max_pooling_size, dropout_rate: float, 
                 activation: str = "relu", *args, **kwargs):
        super(BaseLine, self).__init__(name = name, *args, **kwargs)

        self.model = Sequential()
    
    def conv2dLayer(self, conv_layers: int, conv_filters: int, conv_kernel: int,
                    max_pooling_size: int, dropout_rate: float, 
                    activation: str = "relu"):
        
        for i in range(conv_layers):
            self.model.add(
                Conv2D(filters = conv, kernel_size = conv_kernel, activation = activation)
            )
            self.model.add(
                Dropout(rate = dropout_rate)
            )
            self.model.add(
                MaxPool2D(pool_size = max_pooling_size)
            )
        

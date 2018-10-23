import tensorflow as tf

def convLayer(x, filter_height, filter_width, 
    num_filters, name, stride=1, padding='SAME'):

    """Create Convolutional Layer"""
    
    #Get number of input channels.
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:

        #Create tf variables for the weights and biases of the conv layer.
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        
        #Perform convolution and add bias.
        conv = tf.nn.conv2d(x, Wm strides=[1, stride, stride, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        
        #Add batch-norm layer.
        batch_norm = tf.layers.batch_normalization(z, axis=1, beta_initializer=tf.constant_initializer(0.0),
            gamma_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        out = tf.nn.relu(batch_norm)

        return out

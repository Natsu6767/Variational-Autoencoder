import tensorflow as tf

def convLayer(x, filter_height, filter_width, 
    num_filters, name, stride=2, padding='SAME'):

    """Create Convolutional Layer"""
    
    #Get number of input channels.
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:

        #Create tf variables for the weights and biases of the conv layer.
        W = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels, num_filters],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[num_filters], initializer=tf.constant_initializer(0.0))
        
        #Perform convolution and add bias.
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        z = tf.nn.bias_add(conv, b)
        
        #Add batch-norm layer.
        batch_norm = tf.layers.batch_normalization(z, axis=1, beta_initializer=tf.constant_initializer(0.0),
            gamma_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        out = tf.nn.relu(batch_norm)

    return out

def tConvLayer(x, filter_height, filter_width,
    output_shape, name, stride=2, paddin='SAME'):

    """Create Transposed Convolutional Layer"""
    
    #Get number of input channels.
    input_channels = int(x.get_shape()[-1])

    with tf.variable_scope(name) as scope:
        
        #Create tf variables for the weight and biases of the conv layer.
        W = tf.get_variable('weights', shape=[filter_height, filter_width, output_shape[-1], input_channels],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[output_shape[-1]], initializer=tf.constant_initializer(0.0))
        
        #Perform transposed convolution and add bias.
        tconv = tf.nn.conv2d_transpose(x, w, strides=[1, stride, stride, 1], padding=padding, output_shape=output_shape)
        z = tf.nn.bias_add(tconv, b)
        
        #Add batch-norm layer
        batch_norm = tf.layers.batch_normalization(z, axis=1, beta_initializer=tf.constant_initializer(0.0),
            gamma_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        out = tf.nn.relu(batch_norm)

    return out



def fcLayer(x, output_size, name, relu=True):

    """Create Fully Connected Layer"""
    with tf.variable_scope(name) as scope:
        #Create tf variables for the weight and biases of the fc layer.
        W = tf.get_variable('weights', shape=[x.get_shape()[1], output_size],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

        b = tf.get_variable('biases', shape=[output_size], initializer=tf.constant_initializer(0.0))
        
        #z = Wx + b
        z = tf.nn.bias_add(tf.matmul(x, W), b)

        if relu:
            a = tf.nn.relu(z)
            return a
        else:
            return z
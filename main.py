import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data

#Loading MNIST data.
mnist = input_data.read_data_sets("MNIST_data")

tf.reset_default_graph()
X_in = tf.placeholder(tf.float32, [None, 784])

#Getting the VAE model.
model = VAE(x=X_in, n_z=2, batch_size=128)
optimiser = tf.train.AdamOptimizer(0.001).minimize(model.cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

steps = 10001
for i in range(steps):
    batch = mnist.train.next_batch(batch_size=model.batch_size)[0]

    sess.run(optimiser, feed_dict={X_in : batch})

    #Every epoch.
    if not i % 500:
        loss, gen_image, recon, latent = sess.run([model.cost, model.generated_images, 
            model.reconstruction_loss, model.latent_loss], feed_dict={X_in : batch})
        
        epoch = i//500
        print("Epoch %d:\n" %epoch)
        print("Total Loss: %f\nReconstruction Loss: %f\nLatent Loss: %f" %(loss, np.mean(recon), np.mean(latent))) #Print losses.
        
        #Show input and reconstructions.
        """
        plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
        plt.show()
        plt.imshow(np.reshape(gen_image[0], [28, 28]), cmap='gray')
        plt.show()
        """
        #Visualize 2D latent manifold.
        if epoch in (5, 10, 15, 20) and model.n_z == 2: #Only applicable for 2D latent space as higher dimensions are not visualizable!
            n = 20 #figure with 20x20 digits.
            digit_size = 28
            figure = np.zeros((digit_size*n, digit_size*n))

            #Contruct grid of latent variable values.
            grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
            grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

            #Decode for each square in the grid.
            for i, xi in enumerate(grid_x):
                for j, yj in enumerate(grid_y):
                    z_sample = np.array([xi, yj])
                    z_sample = np.tile(z_sample, model.batch_size).reshape(model.batch_size, 2)
                    x_decoded = sess.run(model.generated_images, feed_dict = {model.z_generated : z_sample})
                    digit = np.reshape(x_decoded[0], [28, 28])
                    figure[i * digit_size: (i+1) * digit_size,
                    j * digit_size: (j+1) * digit_size] = digit

            plt.figure(figsize=(10,10))
            plt.imshow(figure, cmap='bone')
            plt.show()

        print("_"*25)

#Visualization of the Latent Space.
#Only 2 dimensions of the latent space are visualized. Higher dimensions are hard to visualize!
plt_batch = mnist.train.next_batch(batch_size=5000)
z_mu = sess.run(model.z_mean, feed_dict={X_in: plt_batch[0]}) #Get mean encodings (z_mean).
plt.figure(figsize=(8, 6)) 
plt.scatter(z_mu[:, 0], z_mu[:, 1], c=plt_batch[1], cmap='brg')
plt.colorbar()
plt.grid()
plt.show()

#Code for generating new unseen samples.
"""
randoms = [np.random.normal(0, 1, model.n_z) for _ in range(128)]
imgs = sess.run(model.generated_images, feed_dict = {model.z_generated : randoms})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]
for img in imgs[:10]:
    plt.imshow(img, cmap = 'gray')
    plt.show()
"""
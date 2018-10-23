import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data

#Loading MNIST data.
mnist = input_data.read_data_sets("MNIST_data")

tf.reset_default_graph()
X_in = tf.placeholder(tf.float32, [None, 784])

#Getting the VAE model.
model = VAE(x = X_in, n_z = 20, batch_size=64)
optimiser = tf.train.AdamOptimizer(0.001).minimize(model.cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
	batch = mnist.train.next_batch(batch_size=model.batch_size)[0]

	sess.run(optimiser, feed_dict={X_in : batch})

	#Visualize the reconstruction and calculate the losses.
	if not i % 200:
		loss, gen_image, recon, latent = sess.run([model.cost, model.generated_images, 
			model.reconstruction_loss, model.latent_loss], feed_dict={X_in : batch})

		plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
		plt.show()
		plt.imshow(np.reshape(gen_image[0], [28, 28]), cmap='gray')
		plt.show()
		print("Total Loss: %f\nReconstruction Loss: %f\nLatent Loss: %f" %(loss, np.mean(recon), np.mean(latent)))

#To generate new unseen samples.
randoms = [np.random.normal(0, 1, model.n_z) for _ in range(64)]
imgs = sess.run(model.generated_images, feed_dict = {model.z_generated : randoms})
imgs = [np.reshape(imgs[i], [28, 28]) for i in range(len(imgs))]

for img in imgs[:10]:
    plt.imshow(img, cmap = 'gray')
    plt.show()
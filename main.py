import tensorflow as tf

from vae import VAE
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data")

tf.reset_default_graph()
batch_size = 64

X_in = tf.placeholder(tf.float32, [None, 784])

model = VAE(x = X_in, n_z = 20)

optimiser = tf.train.AdamOptimizer(0.001).minimize(model.cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(3000):
	batch = mnist.train.next_batch(batch_size=batch_size)[0]

	sess.run(optimiser, feed_dict={X_in : batch})

	if not i % 500:
		loss, gen_image, recon, latent = sess.run([model.cost, model.generated_images, 
			model.reconstruction_loss, model.latent_loss], feed_dict={X_in : batch})

		plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
		plt.show()
		plt.imshow(gen_image[0], cmap='gray')
		plt.show()
		print("Total Loss: %f\nReconstruction Loss: %f\nLatent Loss: %f" %(loss, recon, latent)
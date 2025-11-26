import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(ckpt_path):
    '''Load parameters
    input:
        ckpt_path(str): save directory
    output:
        U, V(ndarray): latent factors
    '''
    # Create variable placeholder for loading
    checkpoint = tf.train.Checkpoint()
    status = checkpoint.read(ckpt_path)

    # Get variables from checkpoint
    U = checkpoint.U.numpy()
    V = checkpoint.V.numpy()

    return U.T, V.T

def linear_mapping(input_Vs, input_Vt, beta, learning_rate, training_epochs, display_step=100):
    '''Linear mapping function
    input:
        input_Vs(ndarray): source domain matrix
        input_Vt(ndarray): target domain matrix
        beta(float): regularization parameter
        learning_rate(float): learning rate
        training_epochs(int): maximum number of iterations
        display_step(int): display step
    output:
        M, b: mapping function parameters
    '''
    k, m = np.shape(input_Vs)

    # 1. Set variables
    M = tf.Variable(tf.random.normal([k, k], mean=0, stddev=0.1), name="M")
    b = tf.Variable(tf.zeros([m]), name="b")

    optimizer = tf.optimizers.SGD(learning_rate)

    # Convert to tensor
    Vs = tf.constant(input_Vs, dtype=tf.float32)
    Vt = tf.constant(input_Vt, dtype=tf.float32)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # 2. Build model
            predVt = tf.matmul(M, Vs) + tf.expand_dims(b, 1)
            regM = beta * tf.reduce_sum(tf.square(M))
            cost = tf.reduce_mean(tf.square(Vt - predVt)) + regM

        gradients = tape.gradient(cost, [M, b])
        optimizer.apply_gradients(zip(gradients, [M, b]))
        return cost

    # 3. Start training
    for epoch in range(training_epochs):
        cost_val = train_step()

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_val.numpy()))

    # Print variables
    print("Variable: M")
    print("Shape:", M.shape)
    print(M.numpy())
    print("Variable: b")
    print("Shape:", b.shape)
    print(b.numpy())

    # Save model
    checkpoint = tf.train.Checkpoint(M=M, b=b)
    checkpoint.save("model/lm/lm")
    print("Optimization Finished!")

if __name__ == "__main__":
    Us, Vs = load_data("model/mf_s/s")
    # print(np.shape(Vs))
    Ut, Vt = load_data("model/mf_t/t")

    beta = 0.001
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 10
    linear_mapping(Vs, Vt, beta, learning_rate, training_epochs, display_step)

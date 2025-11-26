import os
import numpy as np
import tensorflow as tf

from LM import load_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def MLP(input_Vs, input_Vt, beta, learning_rate, training_epochs, display_step=100):
    '''Multi-layer perceptron mapping function
    input:
        input_Vs(ndarray): source domain matrix
        input_Vt(ndarray): target domain matrix
        beta(float): regularization parameter
        learning_rate(float): learning rate
        training_epochs(int): maximum number of iterations
    output:
        U, V: factorized matrices
    '''

    k, m = np.shape(input_Vs)

    # 1. Initialize parameters
    w1 = tf.Variable(tf.random.truncated_normal([2 * k, k], stddev=0.1), name="w1")
    b1 = tf.Variable(tf.zeros([2 * k, 1]), name="b1")

    w2 = tf.Variable(tf.zeros([k, 2 * k]), name="w2")
    b2 = tf.Variable(tf.zeros([k, 1]), name="b2")

    optimizer = tf.optimizers.Adagrad(learning_rate)

    # Convert to tensor
    Vs = tf.constant(input_Vs, dtype=tf.float32)
    Vt = tf.constant(input_Vt, dtype=tf.float32)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # 2. Build model
            hidden1 = tf.nn.tanh(tf.matmul(w1, Vs) + b1)

            reg_w1 = beta * tf.reduce_sum(tf.square(w1))
            reg_w2 = beta * tf.reduce_sum(tf.square(w2))

            pred = tf.nn.sigmoid(tf.matmul(w2, hidden1) + b2)
            cost = tf.reduce_mean(tf.square(Vt - pred)) + reg_w1 + reg_w2

        gradients = tape.gradient(cost, [w1, b1, w2, b2])
        optimizer.apply_gradients(zip(gradients, [w1, b1, w2, b2]))
        return cost

    # 3. Start training
    for epoch in range(training_epochs):
        cost_val = train_step()

        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_val.numpy()))

    # Print variables
    print("Variable: w1")
    print("Shape:", w1.shape)
    print(w1.numpy())
    print("Variable: b1")
    print("Shape:", b1.shape)
    print(b1.numpy())
    print("Variable: w2")
    print("Shape:", w2.shape)
    print(w2.numpy())
    print("Variable: b2")
    print("Shape:", b2.shape)
    print(b2.numpy())

    # Save model
    checkpoint = tf.train.Checkpoint(w1=w1, b1=b1, w2=w2, b2=b2)
    checkpoint.save("model/mlp/mlp")
    print("Optimization Finished!")
    
if __name__ == "__main__":
    Us, Vs = load_data("model/mf_s/s")
    # print(np.shape(Vs))
    Ut, Vt = load_data("model/mf_t/t")

    beta = 0.001
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 10
    MLP(Vs, Vt, beta, learning_rate, training_epochs, display_step)
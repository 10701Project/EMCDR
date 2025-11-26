import os
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(file_path):
    '''Load data
    input: file_path(string): rating data file path
    output: data(): rating matrix
    '''
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    return data.values

def MF(data, k, learning_rate, beta, training_epochs, display_step=10):
    '''Perform matrix factorization using gradient descent
    input:
        data: rating matrix
        k(int): parameter for matrix factorization
        learning_rate(float): learning rate
        beta(float): regularization parameter
        training_epochs(int): maximum number of iterations
    output:
        U, V: factorized matrices
    '''
    m, n = np.shape(data)

    # 2. Initialize U and V
    U = tf.Variable(tf.random.normal([m, k], mean=0, stddev=0.1), name="U")
    V = tf.Variable(tf.random.normal([n, k], mean=0, stddev=0.1), name="V")

    optimizer = tf.optimizers.SGD(learning_rate)

    # Convert data to tensor
    R = tf.constant(data, dtype=tf.float32)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            # 3. Build model
            pred = tf.matmul(U, tf.transpose(V))

            # L2 regularization
            regU = beta * tf.reduce_sum(tf.square(U))
            regV = beta * tf.reduce_sum(tf.square(V))

            cost = tf.reduce_mean(tf.square(R - pred)) + regU + regV

        gradients = tape.gradient(cost, [U, V])
        optimizer.apply_gradients(zip(gradients, [U, V]))
        return cost

    # 4. Start training
    for epoch in range(training_epochs):
        cost_val = train_step()

        # Print cost
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_val.numpy()))

    # Print variables
    print("Variable: U")
    print("Shape:", U.shape)
    print(U.numpy())
    print("Variable: V")
    print("Shape:", V.shape)
    print(V.numpy())

    # Save model
    checkpoint = tf.train.Checkpoint(U=U, V=V)
    checkpoint.save("model/mf_t/t")
    print("Optimization Finished!")

if __name__ == "__main__":
    data = load_data("data/t_rate.csv")
    MF(data, 5, 0.0002, 0.02, 50000, 1000)

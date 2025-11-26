import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(file_path):
    '''Load data
    input: file_path(string): rating data file path
    output:
        users(list): user list
        movies(list): movie list
        user_ratings(dict): [user]-[list of watched movies] dictionary
    '''
    data = pd.read_csv(file_path, index_col=0)
    data = data.fillna(0)
    users = data.index
    movies = data.columns
    
    user_ratings = defaultdict(dict)
    no = 0
    for i in range(len(users)):
#         print(list(np.where(data.loc[user] != 0.0)[0]))
        user_movies = list(np.where(data.loc[users[i]] != 0.0)[0])
        if len(user_movies) > 1:
            user_ratings[no] = user_movies
            no += 1
    
    movies = [int(movie) for movie in movies]
    return users, movies, user_ratings

def generate_test(user_ratings):
    '''Randomly sample a rated movie i to generate test dataset
    input:
        user_ratings(dict): [user]-[list of watched movies] dictionary
    output:
        user_ratings_test(dict): [user]-[one watched movie i] dictionary
    '''
    user_ratings_test = {}
    for user in user_ratings:
        user_ratings_test[user] = random.sample(user_ratings[user], 1)[0]
    return user_ratings_test

def generate_train_batch(user_ratings, user_ratings_test, n, batch_size=512):
    '''Construct training triplets
    input:
        user_ratings(dict): [user]-[list of watched movies] dictionary
        user_ratings_test(dict): [user]-[one watched movie i] dictionary
        n(int): number of movies
        batch_size(int): batch size
    output:
        train_batch: training batch
    '''
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(), 1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]

        j = random.randint(0, n-1)
        while j in user_ratings[u]:
            j = random.randint(0, n-1)
        
        t.append([u, i, j])
    
    train_batch = np.asarray(t)
    return train_batch

def generate_test_batch(user_ratings, user_ratings_test, n):
    '''Construct test triplets
    input:
        user_ratings(dict): [user]-[list of watched movies] dictionary
        user_ratings_test(dict): [user]-[one watched movie i] dictionary
        movies(list): movie ID list
    output:
        test_batch: test batch
    '''
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(n):
            if j not in user_ratings[u]:
                t.append([u, i, j])
        # print(t)
        yield np.asarray(t)

def BPR(user_ratings, user_ratings_test, movies, k, beta, learning_rate, training_epochs, display_step=10):
    '''Solve BPR using gradient descent
    input:
        user_ratings(dict): [user]-[list of watched movies] dictionary
        user_ratings_test(dict): [user]-[one watched movie i] dictionary
        movies(list): movie ID list
        k(int): parameter for matrix factorization
        beta(float): regularization parameter
        learning_rate(float): learning rate
        training_epochs(int): maximum number of iterations
    output:
        U, V: factorized matrices
    '''

    m = len(user_ratings)
    n = len(movies)

    # 1. Initialize variables
    U = tf.Variable(tf.random.normal([m, k], mean=0, stddev=0.1), name="U")
    V = tf.Variable(tf.random.normal([n, k], mean=0, stddev=0.1), name="V")

    optimizer = tf.optimizers.SGD(learning_rate)

    @tf.function
    def train_step(u_idx, i_idx, j_idx):
        with tf.GradientTape() as tape:
            u_emb = tf.nn.embedding_lookup(U, u_idx)
            i_emb = tf.nn.embedding_lookup(V, i_idx)
            j_emb = tf.nn.embedding_lookup(V, j_idx)

            pred = tf.reduce_mean(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)

            # L2 regularization
            regu = beta * tf.reduce_sum(tf.square(u_emb))
            regi = beta * tf.reduce_sum(tf.square(i_emb))
            regj = beta * tf.reduce_sum(tf.square(j_emb))

            cost = regu + regi + regj - tf.reduce_mean(tf.math.log(tf.sigmoid(pred)))

        gradients = tape.gradient(cost, [U, V])
        optimizer.apply_gradients(zip(gradients, [U, V]))

        auc = tf.reduce_mean(tf.cast(pred > 0, tf.float32))
        return cost, auc

    # 4. Start training
    for epoch in range(training_epochs):
        avg_cost = 0
        for p in range(1, 100):
            uij = generate_train_batch(user_ratings, user_ratings_test, n)
            batch_cost, _ = train_step(uij[:, 0], uij[:, 1], uij[:, 2])
            avg_cost += batch_cost

        # Print cost
        if (epoch + 1) % display_step == 0:
            # Calculate accuracy
            user_count = 0
            avg_auc = 0

            for t_uij in generate_test_batch(user_ratings, user_ratings_test, n):
                test_batch_cost, test_batch_auc = train_step(t_uij[:, 0], t_uij[:, 1], t_uij[:, 2])
                user_count += 1
                avg_auc += test_batch_auc
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost / p), "test_cost", "{:.9f}".format(test_batch_cost.numpy()), "test_auc", "{:.9f}".format(avg_auc / user_count))

    # Print variables
    print("Variable: U")
    print("Shape:", U.shape)
    print(U.numpy())
    print("Variable: V")
    print("Shape:", V.shape)
    print(V.numpy())

    # Save model
    checkpoint = tf.train.Checkpoint(U=U, V=V)
    checkpoint.save("model/bpr_t/t")
    print("Optimization Finished!")

if __name__ == "__main__":
    users, movies, user_ratings = load_data("data/t_rate.csv")
    user_ratings_test = generate_test(user_ratings)

    # Parameters
    k = 20 
    beta = 0.0001 
    learning_rate = 0.01
    training_epochs = 100
    display_step = 10

    BPR(user_ratings, user_ratings_test, movies, k, beta, learning_rate, training_epochs, display_step)
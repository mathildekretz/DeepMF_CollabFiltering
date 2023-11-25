import numpy as np
import os
from tqdm import tqdm, trange
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                        help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()

    # Open Ratings table
    print('Ratings loading...')
    table = np.load(args.name)  ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')


    ### Any method you want ###

    def matrix_factorization(train, ridge_i, ridge_j, k, max_iterations=1500,
                             learning_rate_i=1e-4, learning_rate_j=1e-3):

        #Gradient calculation function
        def gradient(I, U, ridge_i, ridge_j):
            ecart = train - np.dot(I, U.T)
            ecart = np.nan_to_num(ecart, nan=0)
            return -np.dot(ecart, U) + ridge_i * I, -np.dot(ecart.T, I) + ridge_j * U

        # Calculate the mean and standard deviation of the non-NaN values in R
        non_nan_values = train[~np.isnan(train)]
        mean_train = np.mean(non_nan_values)
        std_dev_train = np.std(non_nan_values)

        # Define the positive range for random initialization
        min_value = max(0, mean_train - std_dev_train)
        max_value = mean_train + std_dev_train

        # Initialize random matrices I and U with appropriate dimensions
        # Use a uniform distribution over the positive range
        np.random.seed(1000)
        I = np.random.uniform(min_value, max_value, size=(np.shape(train)[0], k))
        U = np.random.uniform(min_value, max_value, size=(np.shape(train)[1], k))

        # Initialize variables
        iteration = 0

        gradient_I, gradient_U = gradient(I, U, ridge_i, ridge_j)

        # Gradient descent using a while loop with learning rate scheduling
        while iteration < max_iterations and not np.isnan(np.linalg.norm(gradient_I)) and not np.isnan(
                np.linalg.norm(gradient_U)):
            gradient_I, gradient_U = gradient(I, U, ridge_i, ridge_j)

            # Update I and U using the current learning rate
            I -= learning_rate_i * gradient_I
            U -= learning_rate_j * gradient_U

            iteration += 1

            print(f"iteration : {iteration}, "
                  f"norm(grad_I)={np.linalg.norm(gradient_I)}, "
                  f"norm(grad_U)={np.linalg.norm(gradient_U)}")

        # Calculate the predicted matrix after the gradient descent
        R_pred = np.dot(I, U.T)

        # Return the final I and U matrices, as well as the predicted matrix R_pred
        return R_pred


    def deep_mf(r):

        E_0 = matrix_factorization(r, ridge_i=0.01, ridge_j=1, k=1, max_iterations=500,
                                   learning_rate_i=1e-4, learning_rate_j=1e-3)
        E_1 = r - E_0
        P1Q1 = matrix_factorization(E_1, ridge_i=1, ridge_j=1, k=1, max_iterations=500,
                                    learning_rate_i=1e-3, learning_rate_j=1e-2)
        E_2 = E_1 - P1Q1
        P2Q2 = matrix_factorization(E_2, ridge_i=1, ridge_j=1, k=1, max_iterations=500,
                                    learning_rate_i=1e-3, learning_rate_j=1e-2)
        E_3 = E_2 - P2Q2
        P3Q3 = matrix_factorization(E_3, ridge_i=1, ridge_j=1, k=1, max_iterations=500,
                                    learning_rate_i=1e-2, learning_rate_j=1e-2)

        r_pred = E_0 + P1Q1 + P2Q2 + P3Q3
        
        non_nan_indices = ~np.isnan(r)
        r_pred[non_nan_indices] = r[non_nan_indices]

        return np.round(r_pred,1)


    table = deep_mf(table)

    # Save the completed table
    np.save("output.npy", table)  ## DO NOT CHANGE THIS LINE

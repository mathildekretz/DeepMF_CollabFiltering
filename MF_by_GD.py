import numpy as np
import matplotlib.pyplot as plt

#Data loading
train = np.load("ratings_train.npy")
test = np.load("ratings_test.npy")


# Calculate the RMSE loss using non-NaN values in R
def calculate_rmse_loss(test, R):
    #Searching for non-missing value clues    
    non_nan_indices = ~np.isnan(test)

    #Calculate the RMSE
    squared_error = (R[non_nan_indices] - test[non_nan_indices]) ** 2
    rmse_loss = np.sqrt(np.mean(squared_error))

    return rmse_loss


#Implementation of the matrix factorization algorithm by gradient descent
def matrix_factorization(train,k,ridge_i,ridge_j, max_iterations=1000, initial_learning_rate_i=1e-4, initial_learning_rate_j=1e-3, decay_rate=1, decay_steps=100) :   

    #Gradient calculation function
    def gradiant(I,U,ridge_i,ridge_j):
        ecart = train - np.dot(I,U.T) #Calculation of the error
        ecart = np.nan_to_num(ecart, nan=0) # As in the next line it's a simple sum, it's the same as replacing missing values with zeros and summing, or taking only non-missing values in the sum, and we think it's more efficient to use a matrix product with numpy than a rather restrictive for loop.
        return -2*np.dot(ecart,U) + ridge_i*I, -2*np.dot(ecart.T,I) + ridge_j*U #Final calculation of the two gradients 
    
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
    learning_rate_i = initial_learning_rate_i  # Initial learning rate
    learning_rate_j = initial_learning_rate_j  # Initial learning rate
    gradient_I,gradient_U = gradiant(I,U,ridge_i,ridge_j)
    RMSEs=[]
    iterations= []

    # Gradient descent using a while loop with learning rate scheduling
    while iteration < max_iterations and np.linalg.norm(gradient_U)!=float('inf') and np.linalg.norm(gradient_I)!=float('inf') :

        # Compute the gradients for I and U
        gradient_I,gradient_U = gradiant(I,U,ridge_i,ridge_j)

        # Update I and U using the current learning rate
        I -= learning_rate_i * gradient_I
        U -= learning_rate_j * gradient_U

        # Calculate the RMSE loss
        rmse_loss = calculate_rmse_loss(test, np.dot(I, U.T))
        print(f"Iteration {iteration + 1}: RMSE Loss = {round(rmse_loss,4)}, gradient i :{round(np.linalg.norm(gradient_I),2)}, gradient u :{round(np.linalg.norm(gradient_U),2)}")
        RMSEs.append(rmse_loss)
        iterations.append(iteration)

        iteration += 1
        

        # Apply learning rate decay :
        if iteration % decay_steps == 0:
            learning_rate_i *= decay_rate
            learning_rate_j *= decay_rate


    #RMSE display as a function of the number of iterations
    plt.plot(iterations, RMSEs, marker='+', linestyle='-', color='b', label='RMSE')
    plt.legend(loc='upper center')
    plt.xlabel("Number of iterations")
    plt.ylabel('RMSE')
    plt.show()



    # Calculate the predicted matrix after the gradient descent
    R_pred = np.dot(I, U.T)
    non_nan_indices = ~np.isnan(train)
    R_pred[non_nan_indices] = train[non_nan_indices]


    # Return the final I and U matrices, as well as the predicted matrix R_pred
    return R_pred


#Definition of ridge penalty coefficients 
ridge_i = 1
ridge_j = 1e-1

R = matrix_factorization(train, # training matrix
                        k=1, #rank of I and U
                        ridge_i=ridge_i, #ridge penalty coefficient for I 
                        ridge_j=ridge_j, #ridge penalty coefficient for U
                        max_iterations=600, #Maximal number of iteration in for loop of the gradient descent  
                        initial_learning_rate_i=1*10**(-4), #initial learning rate for I
                        initial_learning_rate_j=1*10**(-3), #initial learning rate for I
                        decay_rate=1, # decay rate for the scheduling
                        decay_steps=100) #decay step for the scheduling 

#Rounding of predicted ratings
R=np.round(R,1) 

#Calculation of final RMSE and accuracy
non_nan_indices = ~np.isnan(test)
squared_error = (R[non_nan_indices] - test[non_nan_indices]) ** 2
accuracy = round((1- (np.count_nonzero(squared_error)/np.shape(squared_error)[0]))*100,2)
rmse_loss = round(np.sqrt(np.mean(squared_error)),5)

#RMSE and accuracy display
print(f"RMSE : {rmse_loss},\n Accuracy : {accuracy}")

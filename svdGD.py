import numpy as np
import copy
from tqdm import tqdm

table_genre = np.load('namesngenre.npy')
train = np.load('ratings_train.npy')
test = np.load('ratings_test.npy')


def matrix_factorization(train, epsilon,k,ridge_i,ridge_j, max_iterations=1000, initial_learning_rate_i=1e-4,initial_learning_rate_j=1e-4, decay_rate=0.95, decay_steps=100):
    
    def gradiant(I,U,ridge_i,ridge_j):
        ecart= train - np.dot(I,U.T)
        ecart = np.nan_to_num(ecart, nan=0)
        return -2*np.dot(ecart,U)+2*ridge_i*I,-2*np.dot(ecart.T,I) + 2*ridge_j*U
    
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

    # Define a function to calculate the RMSE loss
    def calculate_rmse_loss(test, I, U):
        # Calculate the predicted matrix
        predicted = np.dot(I, U.T)

        # Calculate the RMSE loss using non-NaN values in R
        non_nan_indices = ~np.isnan(test)
        squared_error = (predicted[non_nan_indices] - test[non_nan_indices]) ** 2
        rmse_loss = np.sqrt(np.mean(squared_error))

        return rmse_loss

    # Initialize variables
    iteration = 0
    learning_rate_i = initial_learning_rate_i  # Initial learning rate
    learning_rate_j = initial_learning_rate_j  # Initial learning rate

    gradient_I,gradient_U = gradiant(I,U,ridge_i,ridge_j)

    # Gradient descent using a while loop with learning rate scheduling
    while iteration < max_iterations and (np.linalg.norm(gradient_I)>epsilon and np.linalg.norm(gradient_U)>epsilon) :


        # Compute the gradients for I and U
        gradient_I,gradient_U = gradiant(I,U,ridge_i,ridge_j)

        # Update I and U using the current learning rate
        I -= learning_rate_i * gradient_I
        U -= learning_rate_j * gradient_U

        # Calculate the RMSE loss
        rmse_loss = calculate_rmse_loss(test, I, U)
        print(f"Iteration {iteration + 1}: RMSE Loss = {round(rmse_loss,4)}, gradient i :{round(np.linalg.norm(gradient_I),2)}, gradient u :{round(np.linalg.norm(gradient_U),2)}")

        iteration += 1
        

        # Apply learning rate decay
        if iteration % decay_steps == 0:
            learning_rate_i *= decay_rate
            learning_rate_j *= decay_rate

    # Calculate the predicted matrix after the gradient descent
    R_pred = np.dot(I, U.T)

    # Return the final I and U matrices, as well as the predicted matrix R_pred
    return R_pred


def representation(rating_mat) :
    A = copy.deepcopy(rating_mat)
    (n, m) = np.shape(rating_mat)

    R_MF=np.round(matrix_factorization(train=A,
                                epsilon =30,
                                k=1,
                                ridge_i=10**(0),
                                ridge_j=10**(-1),
                                max_iterations=10000,
                                initial_learning_rate_i=2*10**(-4),
                                initial_learning_rate_j=2*10**(-3),
                                decay_rate=0.9,
                                decay_steps=100))
    
    non_nan_indices = ~np.isnan(rating_mat)
    R_MF[non_nan_indices]=A[non_nan_indices]
    print(R_MF)
    return np.round(R_MF)


def representation2(rating_mat) :
    A = copy.deepcopy(rating_mat)
    (n, m) = np.shape(rating_mat)

    column_avg = []
    #compute columns averages of R
    for j in range (m) : 
        rj_avg, rj_nb = 0,0
        for i in range(n) :
            if not np.isnan(A[i,j]) : 
                rj_avg += A[i,j]
                rj_nb +=1
        column_avg.append(rj_avg/rj_nb) #column average of rating matrix

    for j in range(m) : 
        for i in range(n) :
            if np.isnan(A[i,j]) : 
                A[i,j] = column_avg[j]
    return A,column_avg


def decomposition(A, k) :
    #compute SV decomposition
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    # print(np.shape(U), np.shape(S), np.shape(VT))
    An = U @ np.diag(S) @ VT

    # U, S, and VT are the matrices such that matrix = U.dot(np.diag(S)).dot(VT)
    U_k = U[:, :k]
    S_k = S[:k]
    VT_k = VT[:k, :]
    A_k = U_k.dot(np.diag(S_k)).dot(VT_k)

    return (A_k)


ratingtest = np.array([[5,2, np.nan],
                       [np.nan, 3, np.nan], 
                       [5, 3, 2], 
                       [np.nan, np.nan, 4],
                       [np.nan, 2, 3]])


def pearson_corrcoef(x, y, muu, muv):
    M_inter = np.where(~np.isnan(x) & ~np.isnan(y)) #common rating index

    sum_num = 0
    for k in M_inter[0] :
        sum_num += (x[k]-muu)*(y[k]-muv)
    
    sum_denomu = np.nansum((x-muu)**2)
    sum_denomv = np.nansum((y-muv)**2)
    # if sum_denomv == 0 or sum_denomu == 0 : return 0 #if there are no ratings

    return sum_num/(np.sqrt(sum_denomu)*np.sqrt(sum_denomv))


def jaccard_similarity(x,y) :
    return False


# Compute the correlation matrix between all pairs of users
def neighborhood(rating_mat, similarity_function, h) :
    (n,m) = rating_mat.shape
    correlation_matrix = np.zeros((n, n)) 
    neighborhood = np.zeros((n,h)) 
    means = []
    print("nanmean en cours")
    for u in tqdm(range(n)) : 
        meanu = np.nanmean(rating_mat[u])
        means.append(meanu)
    print("correlation")
    for i in tqdm(range(n)):
        for j in range(i+1, n): #we force the autocorrelation to be null for a user to not select itself in its neighborhood
            # print(similarity_function(rating_mat[i], rating_mat[j]))
            corr_i_j = similarity_function(rating_mat[i], rating_mat[j], means[i], means[j])
            if ~np.isnan(corr_i_j) : #just in case
                correlation_matrix[i,j] = corr_i_j
                correlation_matrix[j,i] = corr_i_j
    print(correlation_matrix)
    print("neighborhood en cours")
    for i in tqdm(range(n)) :
        neighborhood[i] = sorted(range(n), key=lambda j:correlation_matrix[i,j], reverse=True)[:h]
    return correlation_matrix, neighborhood


A = representation(train)


def prediction_gene(rating_mat,A, k,h) :
    #return the prediction matrix (compute everything)
    
    print("decomposition en cours")
    Ak = decomposition(A,k)
    correlation_matrix, neighborhoods = neighborhood(rating_mat, pearson_corrcoef, h)
    (n,m) = rating_mat.shape
    P = copy.deepcopy(rating_mat)

    # print(neighborhoods[114])
    # print(correlation_matrix[114])
    print("prediction en cours")
    for a in tqdm(range(n)) :
        for j in range(m) :
            if np.isnan(rating_mat[a,j]) : #we dont have rating from user a for item j
                p_aj = 0
                sum_num, sum_denom = 0,0
                for i in range(h) :
                    sum_num += Ak[i,j]*correlation_matrix[a,int(neighborhoods[a,i])]
                    sum_denom += abs(correlation_matrix[a,int(neighborhoods[a,i])])
                if sum_denom == 0 :
                    p_aj = Ak[i,j]
                else : p_aj += sum_num/sum_denom
                P[a,j] = p_aj
    return P


def rmse(prediction, test) :
    non_nan_indices = ~np.isnan(test)
    squared_error = (prediction[non_nan_indices] - test[non_nan_indices]) ** 2
    rmse_loss = np.sqrt(np.mean(squared_error))
    return(rmse_loss)


def mae(prediction, test):
    non_nan_indices = ~np.isnan(test)
    return np.mean(np.abs(test[non_nan_indices] - prediction[non_nan_indices]))


rmse_dict = {}
for h in range(25, 51, 5):
    print(h)
    prediction_from_train = prediction_gene(train,A=A, k=1, h=h)
    rmse_current=rmse(prediction_from_train,test)
    rmse_dict[(h)]=rmse_current

print(rmse_dict)
indices_meilleur = min(rmse_dict, key=lambda x:rmse_dict[x])
print("meilleurs paramètres", indices_meilleur, " rmse associé ", rmse_dict[indices_meilleur])

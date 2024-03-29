import numpy as np

# function to generate a random matrix
def generate_random_matrix(n):
    # generate a random matrix
    A = np.random.randint(0, 10, size=(n, n))
    # if A is not invertible, re-generate
    while np.linalg.det(A) == 0:
        A = np.random.randint(0, 10, size=(n, n))
    return A

# function to generate eigen decomposition
def generate_eigenvalues_eigenvectors(A):
    # get eigen values and eigen vectors
    eigen_values, eigen_vectors = np.linalg.eig(A)
    return eigen_values, eigen_vectors

# function to reconstruct a matrix from eigen values and eigen vectors
def reconstruct_matrix(eigen_values, eigen_vectors):
    # reconstruct a matrix
    A = eigen_vectors.dot(np.diag(eigen_values)).dot(np.linalg.inv(eigen_vectors))
    return A

# function to check if two matrices are equal
def check_equality(A, B):
    return np.allclose(A, B)

# function to round values of a matrix
def round(values, decs=0):
    # round values of a matrix
    return np.round(values, decs)

# main function
if __name__ == "__main__":
    # take input n from user
    n = int(input("Enter the value of n: "))

    # generate a random matrix
    A = generate_random_matrix(n)

    # get eigen values, eigen vectors and eigen decomposition
    eigen_values, eigen_vectors = generate_eigenvalues_eigenvectors(A)

    # reconstruct a matrix from eigen values and eigen vectors
    A_recon = reconstruct_matrix(eigen_values, eigen_vectors)
    round(A_recon)

    # print the results
    print("A:", A)
    print("eigen_values:", eigen_values)
    print("eigen_vectors:", eigen_vectors)
    print("A_recon:", A_recon)
    print("A === A_recon ? ----> ", check_equality(A, A_recon))

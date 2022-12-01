import numpy as np

# function to generate a random matrix
def generate_random_matrix(n, m):
    # generate a random matrix
    A = np.random.randint(0, 10, size=(n, m))
    return A

# function to singular value decomposition
def generate_singular_value_decomposition(A):
    # U = left singular vectors
    # D = singular values
    # V = right singular vectors
    U, D, V = np.linalg.svd(A, full_matrices=False)
    return U, D, V

# function to get Moore-Penrose pseudo-inverse built-in
def get_moore_penrose_pseudo_inverse_numpy(A):
    return np.linalg.pinv(A)

# function to get Moore-Penrose pseudo-inverse from SVD
def get_moore_penrose_pseudo_inverse_manual(A):
    U, D, V = generate_singular_value_decomposition(A)
    D = np.diag(D)
    D
    D_pinv = np.linalg.pinv(D)
    return V.T.dot(D_pinv).dot(U.T)

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
    m = int(input("Enter the value of m: "))
    # generate a random matrix
    A = generate_random_matrix(n, m)

    # compute Moore-Penrose pseudo-inverse
    A_pinv_manual = get_moore_penrose_pseudo_inverse_manual(A)
    A_pinv_numpy = get_moore_penrose_pseudo_inverse_numpy(A)

    # print the results
    print("A:", A)
    print("A_pinv_manual:", A_pinv_manual)
    print("A_pinv_numpy:", A_pinv_numpy)
    print("Are A and A_pinv equal?", check_equality(A_pinv_manual, A_pinv_numpy))


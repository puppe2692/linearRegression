import pandas as pd
import matplotlib.pyplot as plt

input_file = "data.csv"
output_file = "teta.txt"

# upload data
try:
    data = pd.read_csv(input_file, dtype=float, sep=',')
except Exception as e:
    exit(e)

X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# normalization
minX = min(X)
maxX = max(X)
minY = min(Y)
maxY = max(Y)

x = (X - minX) / (maxX - minX)
y = (Y - minY) / (maxY - minY)

# init variable for gradient
t0_tmp = 0
t1_tmp = 0
iterations = 1000
alpha = 0.1


def gradient_function(x, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray (m,)): Data, m examples 
      y (ndarray (m,)): target values
      t1 (scalar)    : model parameter t1
      t0 (scalar)    : model parameter t0
    Returns:
      dj_dt1 (scalar): The gradient of the cost w.r.t. the parameter t1
      dj_dt0 (scalar): The gradient of the cost w.r.t. the parameter t0     
    """
    # Number of training examples
    m = len(x)    
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, t0, t1, alpha, iterations, gradient_function):
    for i in range(iterations):
        dj_dt1, dj_dt0 = gradient_function(x, y, t1, t0)
        t1 -= alpha * dj_dt1
        t0 -= alpha * dj_dt0

    return t1, t0


t1_final, t0_final = gradient_descent(x, y, t0_tmp, t1_tmp, alpha,
                                      iterations, gradient_function)

# Output in the file
with open(output_file, 'w') as f:
    f.write(str(t0_final))
    f.write("\n")
    f.write(str(t1_final))


# Added for line
def normalize(lstX, x):
    return (x - min(lstX)) / (max(lstX) - min(lstX))


def denormalize(lstX, x):
    return x * (max(lstX) - min(lstX)) + min(lstX)


# Visu print
lx = [minX, maxX]
ly = []
for j in lx:
    j = t1_final * normalize(X, j) + t0_final
    price = denormalize(Y, j)
    ly.append(price)

plt.scatter(X, Y)
plt.plot(lx, ly, color='red')
plt.show()

# End of visu

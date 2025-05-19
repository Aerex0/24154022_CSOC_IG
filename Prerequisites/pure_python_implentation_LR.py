
import pandas as pd
import time

x_train_df=pd.read_csv('X_train.csv')
y_train_df=pd.read_csv('y_train.csv')

x_val_df=pd.read_csv('X_val.csv')
y_val_df=pd.read_csv('y_val.csv')

x_train = x_train_df.values.tolist()
y_train = y_train_df.values.flatten().tolist()


x_val = x_val_df.values.tolist()
y_val = y_val_df.values.flatten().tolist()


w = [0.0]*len(x_train[0])
b=0



def prediction(x,w,b):
    n=len(x)
    p=0


    for i in range(n):
        p_i =  x[i]*w[i]
        p = p + p_i
    p =p+b


    return p


def compute_cost(x_train, y_train, w, b):
    m = len(x_train)


    total_cost = 0

    for i in range(m):
        f_wb = prediction(x_train[i], w, b)

        total_cost += (f_wb - y_train[i]) ** 2
        

    return total_cost / (2 * m)



def compute_gradient(x,y,w,b):
    m=len(x)
    n=len(x[0])

    dj_dw = [0.0] * n
    dj_db = 0.0

    for i in range(m):
        err = prediction(x[i],w,b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + (err*x[i][j])
        dj_db= dj_db + err
    
    dj_dw = [val / m for val in dj_dw]
    dj_db = dj_db/m

    return dj_dw,dj_db



J_hist_pure= []

def gradient_descent(x_train, y_train, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = w_in[:]
    b = b_in
    J_history = []

    for i in range(num_iters):
        # Compute gradients
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)

        
        for j in range(len(w)):
            w[j] = w[j] - alpha * dj_dw[j]

        b = b - alpha * dj_db

        
        if i < 100000:
            cost = cost_function(x_train, y_train, w, b)
            J_history.append(cost)
            J_hist_pure.append(cost)



        if i % 100 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:.6f}")

    return w, b, J_history





# Before training
# y_train = [val / 1000 for val in y_train]
# y_val = [val / 1000 for val in y_val]


# Start the pure pthon  training
alpha = 0.005
iterations = 1500

start_time = time.time()

final_w, final_b, J_hist = gradient_descent(
    x_train, y_train,
    w, b,
    compute_cost,
    compute_gradient,
    alpha,
    iterations
)

end_time = time.time()



convergence_time = end_time - start_time

print(f"Convergence Time: {convergence_time:.2f} seconds")


#---------------------------------------------

def mean_absolute_error(y_true, y_pred):
    m = len(y_true)
    total_error = 0
    for i in range(m):
        total_error += abs(y_true[i] - y_pred[i])
    return total_error / m

def root_mean_squared_error(y_true, y_pred):
    m = len(y_true)
    total_error = 0
    for i in range(m):
        total_error += (y_true[i] - y_pred[i]) ** 2
    return (total_error / m) ** 0.5

def r_squared(y_true, y_pred):
    m = len(y_true)
    mean_y = sum(y_true) / m
    ss_tot = 0
    ss_res = 0
    for i in range(m):
        ss_tot += (y_true[i] - mean_y) ** 2
        ss_res += (y_true[i] - y_pred[i]) ** 2
    return 1 - (ss_res / ss_tot)

#------------------------------------------------------------------


y_val_pred = [prediction(x, final_w, final_b) for x in x_val]
y_train_pred = [prediction(x, final_w, final_b) for x in x_train]


mae = mean_absolute_error(y_val, y_val_pred)
rmse = root_mean_squared_error(y_val, y_val_pred)
r2 = r_squared(y_val, y_val_pred)

print(f"\nValidation MAE: {mae:.4f}")
print(f"\nValidation RMSE: {rmse:.4f}")
print(f"\nValidation R²: {r2:.4f}")



mae_t = mean_absolute_error(y_train, y_train_pred)
rmse_t = root_mean_squared_error(y_train, y_train_pred)
r2_t = r_squared(y_train, y_train_pred)


print(f"\nTraining MAE: {mae_t:.4f}")
print(f"\nTraining RMSE: {rmse_t:.4f}")
print(f"\nTraining R²: {r2_t:.4f}")



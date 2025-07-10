import numpy as np
import matplotlib.pyplot as plt

W_original = 1.0
B_Original = 0.0
# Known data points
X = np.array([-2, -1, 1, 2], dtype=np.float32)
Y = W_original * X + B_Original # Y = wx + b

# Graph of the known data points
fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.axhline(color='lightgray')
ax.axvline(color='lightgray')
fig.savefig("Original.png")

# model prediction
def forward(w, x, b):
    return w * x + b

# initialize a randon weight
w = 10.0

y_predicted = forward(w, X, B_Original)
# Graph actual vs prediction
fig, ax = plt.subplots()
ax.scatter(X, Y)
ax.plot(X, y_predicted, 'green')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.axhline(color='lightgray')
ax.axvline(color='lightgray')
fig.savefig("Original_vs_predicted.png")

# MSE
def loss(y, y_predicted):
    return ((y - y_predicted)**2).mean()

# Pick a bunch of random weights
W=[-0.5, 0, 0.5, 1.5, 2, 2.5]

# List store the calculated losses
L=[]

# Calculate loss for each w
for w in W: 
    y_predicted = forward(w, X, B_Original)
    L.append(loss(Y, y_predicted))

# Graph loss with respect to weight
fig, ax = plt.subplots()
ax.set_xlabel('weight')
ax.set_ylabel('loss')
ax.set_ylim(-1, 8)
ax.scatter(W, L)
fig.savefig("Lossplot.png")

# Compute tangent slopes using central difference
# Plotting
fig, ax = plt.subplots()
ax.set_xlabel('weight')
ax.set_ylabel('loss')
ax.set_ylim(-1, 8)
ax.scatter(W, L, color='blue', label='Loss Points')

h = 0.01
delta_x = 0.2  # width of tangent line segments
for i, w in enumerate(W):
    # Compute slope (dL/dw)
    L_plus = loss(Y, forward(w + h, X, B_Original))
    L_minus = loss(Y, forward(w - h, X, B_Original))
    slope = (L_plus - L_minus) / (2 * h)
    
    # Define small x-range around the point for the tangent line
    x_start = w - delta_x
    x_end = w + delta_x
    x_vals = np.linspace(x_start, x_end, 10)
    
    # Line equation: y = L[i] + slope * (x - W[i])
    y_vals = L[i] + slope * (x_vals - w)
    
    ax.plot(x_vals, y_vals, color='red', alpha=0.7)


plt.legend()
fig.savefig("Lossplot_with_tangents.png")

# gradient of loss wrt weight
def gradient_dl_dw(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

# List store the calculated losses
L=[]

# List store the calculated gradients
G=[]

# Calculate loss and gradient for each w
for w in W: 
    y_predicted = forward(w, X, B_Original)
    L.append(loss(Y, y_predicted))
    G.append(gradient_dl_dw(X, Y, y_predicted))


fig, ax = plt.subplots()
ax = plt.subplot()
ax.set_xlabel('weight')
ax.set_ylabel('loss')
ax.set_ylim(-1, 8)
ax.scatter(W, L)

# Add gradient labels next to each point
for i, g in enumerate(G):
    plt.text(W[i]+.05, L[i]+.05, g, fontsize=10)
    fig.savefig("LossplotWGradient.png")

def gradient_dl_db(x, y, y_predicted):
    return np.dot(2, y_predicted-y).mean()

# Training
learning_rate = 0.01
epochs = 500

w = 10
b = 10
for epoch in range(epochs):
    # forward pass
    # calculate predictions
    y_predicted = forward(w, X, b)

    # calculate losses
    l = loss(Y, y_predicted)

    # backpropagation
    # calculate gradients
    dw = gradient_dl_dw(X,Y, y_predicted)

    db = gradient_dl_db(X, Y, y_predicted)

    # gradient descent
    # update weights
    w -= learning_rate * dw

    b -= learning_rate * db

    # print info
    if(epoch % 1==0):
        print(f'epoch {epoch+1}: w={w:.3f}, b={b:.3f}, loss={l:0.8f}, dw={dw:.3f}, forward(10)={forward(w,10,b):0.3f}')
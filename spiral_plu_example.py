import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter, Module

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split

from plu_activation import PeriodicLinearUnit

# --- Alternative Activation Functions Definitions ---

class SnakeFunction(Function):
    @staticmethod
    def forward(ctx, x, a):
        # Thank you to EdwardDixon's implementation of snake in PyTorch from here: https://github.com/EdwardDixon/snake
        ctx.save_for_backward(x, a)
        return torch.where(a == 0, x, x + (torch.sin(a * x) ** 2) / a)

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        sin2ax = torch.sin(2 * a * x)
        grad_x = grad_output * (1 + sin2ax)
        grad_a = grad_output * torch.where(a == 0, x**2, (sin2ax * x / a) - (torch.sin(a * x) / a) ** 2)
        return grad_x, grad_a
class Snake(Module):
    def __init__(self, in_features, a=1.0, trainable=True):
        super().__init__()
        initial_a = torch.full((in_features,), float(a))
        if trainable:
            self.a = Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x):
        return SnakeFunction.apply(x, self.a)

# --- Data Generation ---

def create_spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)

# --- Model Definition ---

class SimpleMLP(nn.Module):
    def __init__(self, activation_fn, hidden_size=8):
        super().__init__()
        self.layer1 = nn.Linear(2, hidden_size)
        self.activation = activation_fn
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

# --- Main Script ---

if __name__ == '__main__':
    # Hyperparameters
    N_POINTS = 500
    N_CLASSES = 2
    EPOCHS = 500
    LR = 0.01
    HIDDEN_SIZE = 2

    # Create data
    X, y = create_spiral_data(N_POINTS, N_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate models
    init_rho_alpha = 5.0
    init_rho_beta = 0.15
    models = {
        'ReLU': SimpleMLP(nn.ReLU(), hidden_size=HIDDEN_SIZE),
        'GELU': SimpleMLP(nn.GELU(), hidden_size=HIDDEN_SIZE),
        'Snake': SimpleMLP(Snake(1, a=1.0), hidden_size=HIDDEN_SIZE),
        'PLU': SimpleMLP(PeriodicLinearUnit(init_alpha=1.0, init_beta=1.0, init_rho_alpha=init_rho_alpha, init_rho_beta=init_rho_beta), hidden_size=HIDDEN_SIZE)
    }
    
    optimizers = {name: torch.optim.Adam(model.parameters(), lr=LR) for name, model in models.items()}
    criterion = nn.BCEWithLogitsLoss()

    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle(f'Activation Function Comparison on Spiral Problem (init_rho_alpha={init_rho_alpha}, init_rho_beta={init_rho_beta}, hidden_size={HIDDEN_SIZE})')

    # Create a meshgrid for decision boundary plotting
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

    # Store artists for animation
    contour_artists = [None] * 4

    plu_ax_index = list(models.keys()).index('PLU')
    param_text = axes[plu_ax_index].text(0.02, 0.02, '', transform=axes[plu_ax_index].transAxes, 
                                         fontsize=8, verticalalignment='bottom', 
                                         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))

    def update(epoch):
        # Use a list to collect all modified artists for blitting
        all_modified_artists = []

        for i, (name, model) in enumerate(models.items()):
            # Training step
            model.train()
            optimizer = optimizers[name]
            
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update plot only every 5 epochs to speed up animation
            if epoch % 5 == 0:
                ax = axes[i]
                
                # Remove old contour
                if contour_artists[i] is not None:
                    contour_artists[i].remove()
                
                # Plot new decision boundary
                model.eval()
                with torch.no_grad():
                    Z = model(grid_tensor).reshape(xx.shape)
                    Z_prob = torch.sigmoid(Z)
                
                contour_artists[i] = ax.contourf(xx, yy, Z_prob.numpy(), cmap='coolwarm', alpha=0.8, levels=np.linspace(0, 1, 11))
                ax.set_title(f'{name} - Epoch {epoch}, Loss: {loss.item():.4f}')
                
                all_modified_artists.append(contour_artists[i])

                if name == 'PLU':
                    # Taking the mean of the parameters for display purposes
                    p_alpha = model.activation.alpha.mean().item()
                    p_beta = model.activation.beta.mean().item()
                    p_rho_alpha = model.activation.rho_alpha.mean().item()
                    p_rho_beta = model.activation.rho_beta.mean().item()
                    
                    text_str = (f'α: {p_alpha:.2f}, β: {p_beta:.2f}\n'
                                f'ρ_α: {p_rho_alpha:.2f}, ρ_β: {p_rho_beta:.2f}')
                    param_text.set_text(text_str)
                    all_modified_artists.append(param_text)

        return all_modified_artists


    # Initial plot setup
    for i, name in enumerate(models.keys()):
        axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train.squeeze(), s=10, cmap='coolwarm', edgecolors='k', linewidth=0.5)
        axes[i].set_xlim(xx.min(), xx.max())
        axes[i].set_ylim(yy.min(), yy.max())

    # Create and save the animation
    print("Creating animation... This may take a minute.")
    ani = FuncAnimation(fig, update, frames=range(EPOCHS), blit=True, repeat=False, interval=20)
    
    # Save the animation. You might need ffmpeg installed
    ani.save('spiral_activation_comparison.mp4', writer='ffmpeg', fps=30)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

import matplotlib.pyplot as plt

def acc_loss(title, n_runs, matrix_acc, matrix_loss, test_acc):

    # Define picture
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    # Available colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    curves = ['A-', 'B-', 'C-', 'D-', 'E-', 'F-', 'G-']

    # First subplot LOSS
    axes[0].set_ylabel("Erro", fontsize=14)
    
    # Second subplot ACC
    axes[1].set_ylabel("Acurácia", fontsize=14)
    axes[1].set_xlabel("Época", fontsize=14)

    for i in range(n_runs):
        # Colors range from C0 to C9
        label = curves[i] + str(test_acc[i])
        axes[0].plot(matrix_loss[i][:], color=colors[i], label=label)
        axes[1].plot(matrix_acc[i][:], color=colors[i], label=label)


    axes[0].legend()
    plt.show()

def test_acc(title, test_acc, parameter_name, x_values):

    # Define picture
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    # Axis labels
    axes.set_ylabel("Acurácia no Teste", fontsize=14)
    axes.set_xlabel(parameter_name, fontsize=14)
    axes.set_xscale(x_values)

    # Plot
    axes.plot(x_values, test_acc, color='b', label='A')

    plt.show()


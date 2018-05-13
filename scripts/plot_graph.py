import matplotlib.pyplot as plt

def acc_loss(title, n_runs, matrix_acc, matrix_loss, test_acc, plot_dir, labels=None):

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

    if (labels=None):
        for i in range(n_runs):
            # Colors range from C0 to C9
            label = curves[i] + str(test_acc[i])
            axes[0].plot(matrix_loss[i][:], color=colors[i], label=label)
            axes[1].plot(matrix_acc[i][:], color=colors[i], label=label)
    else:
        for i in range(n_runs):
            # Colors range from C0 to C9
            label = str(labels[i]) + ":" + str(test_acc[i])
            axes[0].plot(matrix_loss[i][:], color=colors[i], label=label)
            axes[1].plot(matrix_acc[i][:], color=colors[i], label=label)


    axes[0].legend()

    # To show graph, uncomment line below
#    plt.show()

    plt.savefig(plot_dir + title + '_ae')

def test_acc(title, test_acc, plot_dir, parameter_name="Número_do_Treino", x_values=None):

    # Define picture
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    # Axis labels
    axes.set_ylabel("Acurácia no Teste", fontsize=14)
    axes.set_xlabel(parameter_name, fontsize=14)

    # Plot
    if (x_values == None):
        axes.plot(test_acc, color='b', label='A')
    else:
        axes.plot(x_values, test_acc, color='b', label='A')

    # To show graph, uncomment line below
#    plt.show()

    plt.savefig(plot_dir + title + '_erro')


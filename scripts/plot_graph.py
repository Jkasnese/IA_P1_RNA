import matplotlib.pyplot as plt

def acc_loss(n_runs, matrix_acc, matrix_loss, test_acc):

    # Define picture
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    # Available colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


    # First subplot LOSS
    axes[0].set_ylabel("Loss", fontsize=14)
    
    # Second subplot ACC
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)

    for i in range(n_runs):
        # Colors range from C0 to C9
        label = str(test_acc[i])
        axes[0].plot(matrix_loss[i][:], color=colors[i], label=label)
        axes[1].plot(matrix_acc[i][:], color=colors[i], label=label)


    axes[0].legend()
    plt.show()


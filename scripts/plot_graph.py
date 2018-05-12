import matplotlib.pyplot as plt

def plot_graph(argc, argv):

    # Ploting graphics
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results, 'r')
    axes[0].plot(train_accuracy_results, 'b')

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results, 'g')
    axes[1].plot(train_loss_results, 'y')

    plt.show()


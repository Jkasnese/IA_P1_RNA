import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
th = np.linspace(0, 2*np.pi, 128)

def demo(sty):
    mpl.style.use(sty)
    fig, ax = plt.subplots(figsize=(3, 3))

    ax.set_title('style: {!r}'.format(sty), color='b')

    ax.plot(th, np.cos(th), color='b', label='C1')
    ax.plot(th, np.sin(th), color='r', label='C2')
    ax.legend()

    plt.show()

def test_acc2(title, test_acc):

    # Define picture
    fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    # Axis labels
    axes.set_ylabel("Acurácia no Teste", fontsize=14)
    axes.set_xlabel("Parâmetro", fontsize=14)

    # Plot
    axes.plot(test_acc, color='b', label='A')

    plt.show()

test_acc = []
for i in range(5):
    test_acc.append(i)

test_acc2("afef", test_acc)

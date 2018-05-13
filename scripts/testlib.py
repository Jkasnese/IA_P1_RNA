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

demo('default')
demo('seaborn')

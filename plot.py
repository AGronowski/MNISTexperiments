import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_4_history(train_loss, train_acc, val_loss, val_acc, big_title='', start=0, end=None):

    # How much to plot, everything by default
    if not end:
        end = len(train_loss)
    train_loss = train_loss[start:end]
    train_acc = train_acc[start:end]
    val_loss = val_loss[start:end]
    val_acc = val_acc[start:end]

    # Necessary so that first epoch is plotted at 1 instead of 0
    dim = np.arange(start + 1, end + 1)

    # Change marker based on amount of data
    marker = 'o--'
    if end - start > 50:
        marker = '--'

    # GridSpec is necessary to prevent suptitle from overlapping
    fig = plt.figure(1)
    gs1 = gridspec.GridSpec(1, 2)
    ax_list = [fig.add_subplot(ss) for ss in gs1]

    # Loss plot on left
    ax_list[0].plot(dim, train_loss, marker, label="Adam")
    ax_list[0].plot(dim, val_loss, marker, label="AMSGrad")
    ax_list[0].set_xlabel("Epoch")
    ax_list[0].set_ylabel("Loss")
    ax_list[0].legend(loc='best')
    ax_list[0].set_title('Cross-Entropy Loss', fontsize=12)

    # Accuracy plot on right
    ax_list[1].plot(dim, train_acc, marker, label="Adam")
    ax_list[1].plot(dim, val_acc, marker, label="AMSGrad")
    ax_list[1].set_xlabel("Epoch")
    ax_list[1].set_ylabel("Accuracy")
    ax_list[1].set_title('Accuracy', fontsize=12)
    ax_list[1].legend(loc='best')

    # Add title
    fig.suptitle(big_title, fontsize=14)
    gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.show()


files_ams = np.load('history_ams.npz')
files_adam = np.load('history_adam.npz')

plt.close()
plot_4_history(files_adam['train_loss'],files_adam['train_acc'],files_ams['train_loss'],files_ams['train_acc'],big_title="Training")
plot_4_history(files_adam['val_loss'],files_adam['val_acc'],files_ams['val_loss'],files_ams['val_acc'],big_title="Testing")
plt.close()
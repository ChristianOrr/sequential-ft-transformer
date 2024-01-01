import numpy as np
import matplotlib.pyplot as plt


def get_model_importances(importances, title="Importances", figsize=(15,7)):
    imps_sorted = importances.mean().sort_values(ascending=False)
    
    plt.figure(figsize=figsize)
    ax = imps_sorted.plot.bar()
    for p in ax.patches:
        ax.annotate(str(np.round(p.get_height(), 4)), (p.get_x(), p.get_height() * 1.01))
    plt.title(title)
    plt.show()
    
    return imps_sorted


def plot_training_hist(hist, explanations, figsize=(20,10)):
    output_layer_name = ""
    if explanations:
        output_layer_name = "output_"
    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=2)

    ax[0].plot(hist.history['loss'], label='Training Loss')
    ax[0].plot(hist.history['val_loss'], label='Validation Loss')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].legend() 

    ax[1].plot(hist.history[f'{output_layer_name}PR AUC'], label='Training PR AUC')
    ax[1].plot(hist.history[f'val_{output_layer_name}PR AUC'], label='Validation PR AUC')
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("PR AUC")
    ax[1].legend()

    fig.suptitle("Training Validation Metrics", fontsize=14)

    return fig
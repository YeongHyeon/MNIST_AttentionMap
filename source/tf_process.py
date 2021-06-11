import os
import scipy.ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import source.utils as utils

def training(agent, dataset, batch_size, epochs):

    print("\n** Training of the CNN to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0

    for epoch in range(epochs):

        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, tt=0)
            if(len(minibatch['x'].shape) == 1): break
            step_dict = agent.step(minibatch=minibatch, iteration=iteration, training=True)
            iteration += 1
            if(minibatch['t']): break

        print("Epoch [%d / %d] | Loss: %f" %(epoch, epochs, step_dict['losses']['entropy']))
        agent.save_params(model='model_0_finepocch')

def test(agent, dataset, batch_size):

    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        agent.load_params(model=path_model)
        utils.make_dir(path=os.path.join(savedir, path_model), refresh=False)

        saveidx = 0
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, tt=1)
            if(len(minibatch['x'].shape) == 1): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            for idx_y, _ in enumerate(minibatch['y']):
                y_true = np.argmax(minibatch['y'][idx_y])
                y_pred = np.argmax(step_dict['y_hat'][idx_y])

                canvas, canvas_attn = \
                    minibatch['x'][idx_y], scipy.ndimage.zoom(step_dict['attn'][idx_y].numpy(), 4, order=3)

                plt.clf()
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 3, 1)
                plt.axis('off')
                plt.title("Input")
                plt.imshow(canvas[:, :, 0], cmap='gray')

                plt.subplot(1, 3, 2)
                plt.axis('off')
                plt.title("Attention Map")
                plt.imshow(canvas_attn[:, :, 0], cmap='jet')

                plt.subplot(1, 3, 3)
                plt.axis('off')
                plt.title("Overlap")
                plt.imshow(canvas[:, :, 0], cmap='gray')
                plt.imshow(canvas_attn[:, :, 0], cmap='jet', alpha=0.5)

                plt.tight_layout()
                plt.savefig(os.path.join(savedir, path_model, "true_%d;pred_%d;%08d.png" %(y_true, y_pred, saveidx)))
                plt.close()
                saveidx += 1

            if(minibatch['t']): break

import os

import numpy as np
import matplotlib.pyplot as plt

def plot_test_image(image, target, predict, count, data_type, model_name) :
    save_root_path = "./example/{}".format(data_type)

    if model_name == 'fcn8s' :
        image = np.mean(image.squeeze().detach().cpu().numpy(), axis=0)
    else :
        image = image.squeeze().detach().cpu().numpy()
    target = target.squeeze().detach().cpu().numpy()
    predict = predict.squeeze().detach().cpu().numpy()
    predict_ = (predict >= 0.5).astype(np.int_)

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(image, cmap='gray')
    ax[0].axis('off'); ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].imshow(target, cmap='gray')
    ax[1].axis('off'); ax[1].set_xticks([]); ax[1].set_yticks([])

    ax[2].imshow(predict, cmap='gray')
    ax[2].axis('off'); ax[2].set_xticks([]); ax[2].set_yticks([])

    ax[3].imshow(predict_, cmap='gray')
    ax[3].axis('off'); ax[3].set_xticks([]); ax[3].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

    if not os.path.exists(os.path.join(save_root_path, "{}".format(model_name))):
        os.makedirs(os.path.join(save_root_path, "{}".format(model_name)))

    plt.savefig(os.path.join(save_root_path, "{}/example_{}.png".format(model_name, str(count))),
                bbox_inches='tight', pad_inches=0)

    plt.close()

def plot_loss(history, save_path) :
    train_loss, test_loss = history['train_loss'], history['val_loss']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color='r')
    plt.plot(np.arange(len(test_loss)), test_loss, label='test loss', color='skyblue')

    plt.legend(loc='upper right')

    plt.savefig(save_path, dpi=300, bbox_inches = 'tight', pad_inches = 0)
    plt.close()
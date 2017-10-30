"""
Train on all MNIST digits and then select the worst ones
to be used as the test set.

Measure accuracy on the results.
"""

import matplotlib.pyplot
import numpy as np

from evilsplit import train_and_test, sample_classes, sample_images

NUM_SAMPLES = 70000

def main():
    """
    Program entry-point.
    """
    print('Training on all data...')
    _, all_losses = train_and_test(np.arange(NUM_SAMPLES), np.arange(NUM_SAMPLES))
    best_to_worst = np.array(next(zip(*sorted(enumerate(all_losses), key=lambda x: x[1]))))
    train_indices, test_indices = even_adversarial_split(best_to_worst)

    print('Training on adversarial split.')
    corrects, _ = train_and_test(train_indices, test_indices)
    print('Mean accuracy: ' + str(np.mean(np.array(corrects).astype('float32'))))

    print('Plotting some samples.')
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    show_image_grid(sample_images(train_indices), figure=1)
    show_image_grid(sample_images(test_indices), figure=2)
    matplotlib.pyplot.show()

def show_image_grid(images, figure=1):
    """
    Show some images in a grid.
    """
    big_image = np.zeros((28*4, 28*4), dtype='float32')
    images = [i.reshape(28, 28) for i in images]
    for row in range(4):
        for col in range(4):
            big_image[row*28 : (row+1)*28, col*28 : (col+1)*28] = images[row*4 + col]
    matplotlib.pyplot.figure(figure)
    matplotlib.pyplot.imshow(big_image)

def even_adversarial_split(best_to_worst):
    """
    Select the 1000 worst samples from each class and turn
    that into the test set.

    Returns (train, test).
    """
    labels = sample_classes(best_to_worst)
    train, test = [], []
    for label in range(10):
        samples = best_to_worst[labels == label]
        train.extend(samples[:-1000])
        test.extend(samples[-1000:])
    return train, test

if __name__ == '__main__':
    main()

# LT2316 H18 Assignment 2

Git project for implementing assignment 2 in [Asad Sayeed's](https://asayeed.github.io) machine learning class in the University of Gothenburg's Masters
of Language Technology programme.

## Your notes
**Running train.py:** python train.py -P 'A' *-m int* '/scratch/gusastam/' 'weights-filename' *-w 'str'* 'model-filename' 'cat1' 'cat2' 'cat3' ...
<br/>\* Cursive shows that the argument is optional.
<br/>\* In -m, 'int' is a placeholder for an integer of choice (for max instances per category).
<br/>\* -w is a decision whether to load weights from checkpointed file or not, and which file ('str') to load from. Keep track of the file you want to load from (from a previous run), so that the categories are the same etc..
<br/>\* 'model-filename' should be the filename + .json .

**Running test.py:** python test.py -P 'A' *-m int* 'model-filename' 'cat1' 'cat2' 'cat3' ...
<br/>\* Cursive shows that the argument is optional.
<br/>\* In -m, 'int' is a placeholder for an integer of choice (for max instances per category).
<br/>\* 'model-filename' should be the name of the model you want to load (as created by train.py).

**Description of Architecture:** This Autoencoder makes use of two Convolutional layers followed by ReLU activation and MaxPooling layers in the encoder. The decoder consists of two Deconvolutional layers followed by ReLU activation and Upsamling layers. At the very end, it has a Deconvolutional layer with Sigmoid activation which finalizes the recreation of the image. I built it this way based on a number of guides and tutorials I found online (credit in train.py), all of which seemed to following a somewhat similar pattern. As for batch size, I went with 32, which appears to be a very common one (even standard according to documentation), and the steps_per_epoch of the fit_generator is also based on that. I was initially using binary_crossentropy as my loss function, but since this left the loss constantly around 0.5, I ended up switching to MSE after asking Asad about it. The number of epochs was initially 30, but I lowered it in favour of being able to actually run the script to its end a few times. Each epoch takes more than a minute with only two categories to run.
A diagram of the architecture is included in the repo.

**Disclaimer:** I did not end up finishing the assignment and I did not run multiple trainings with differences in architecture and hyperparameters. In the end, once I actually started figuring some stuff out and getting stuff to work, I was already late with the submission and wanting to get working on my project properly. Thus I will submit what I did end up doing and have you grade me on that. It should be somewhat more than half of the assignment, if I am not mistaken.

**What did I learn? (and why is it bad to validate/test on training data?):**
All in all, I feel like mostly what this assignment has taught me is basically some coding in Keras, which of course was new to me. I now know how to build a model with layers and perhaps a little bit about what kind of layers might be useful for different things - this mainly based on what I have seen while googling around to get inspiration. I have probably also learned how to solve some problems one might encounter while programming with Keras, since I did run into a bunch along the way.

Validating/testing on training data is bad since you are then testing it on data it has already seen, and that doesn't tell you anything about how the model would do on unseen data. It also means that you risk overfitting, i.e. the model getting too specialised on the training data and then being less generalised and worse in regards to unseen data.

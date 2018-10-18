# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
import mycoco
from keras.layers import Input, Conv2D, Dense, Activation, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras import optimizers


# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    # Checking if maxinstances.
    if args.maxinstances != None:
        idsstart = mycoco.query(args.categories, exclusive=True)
        ids = []
        for id in idsstart:
            ids.append(list(id[:args.maxinstances]))
    else:
        ids = mycoco.query(args.categories, exclusive=True)

    # Gets image data (without labels).
    imgs = mycoco.iter_images_nocat(ids, args.categories, batch=32)

    # Counts number of ids as an approximate number of images for later steps_per_epoch.
    # May vary somewhat due to potentially excluding any black and white images.
    num_imgs = 0
    for id in ids:
        num_imgs += len(id)

    # Model Architecture:
    # input
    inputlayer = Input(shape=(200,200,3))
    # encoder
    conv2dlayer = Conv2D(8, (3,3), padding='same')(inputlayer)
    relulayer = Activation('relu')(conv2dlayer)
    maxpool2dlayer = MaxPooling2D(pool_size=(2,2))(relulayer)
    conv2dlayer2 = Conv2D(16, (3,3), padding='same')(maxpool2dlayer)
    relulayer2 = Activation('relu')(conv2dlayer2)
    maxpool2dlayer2 = MaxPooling2D(pool_size=(2,2))(relulayer2)
    encoded = maxpool2dlayer2
    # decoder
    conv2dtranslayer = Conv2DTranspose(16, (3,3), padding='same')(maxpool2dlayer2)
    relulayer3 = Activation('relu')(conv2dtranslayer)
    upsamplinglayer = UpSampling2D((2,2))(relulayer3)
    conv2dtranslayer2 = Conv2DTranspose(8, (3,3), padding='same')(upsamplinglayer)
    relulayer4 = Activation('relu')(conv2dtranslayer2)
    upsamplinglayer2 = UpSampling2D((2,2))(relulayer4)
    conv2dtranslayer3 = Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same')(upsamplinglayer2)

    # Creating model.
    model = Model(inputlayer, conv2dtranslayer3)
    model.summary()

    # Loading checkpointed weights.
    if args.loadweights != None:
        print('Loading saved weights from file.')
        model.load_weights(args.checkpointdir + args.loadweights)

    # Compiling model.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    # checkpoint
    filepath = args.checkpointdir + args.chkpntfilename
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    steps = round(num_imgs / 32)

    model.fit_generator(imgs, steps_per_epoch=steps, epochs=5, callbacks=callbacks_list)

    # Saving model.
    print("Saving model to file.")
    filename = args.modelfile
    model_json = model.to_json()
    with open(filename, "w") as model_file:
        model_file.write(model_json)

'''
Credit to:
The Keras Blog - https://blog.keras.io/building-autoencoders-in-keras.html
Machinelearningmastery.com - https://machinelearningmastery.com/save-load-keras-deep-learning-models/
https://machinelearningmastery.com/check-point-deep-learning-models-keras/
Tanmay Bakshi on YouTube - https://www.youtube.com/watch?v=6Lfra0Tym4M&feature=youtu.be
for inspiration and help with building my autoencoder, checkpointing, and saving/loading models in Keras.
'''

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('train')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('checkpointdir', type=str,
                        help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('chkpntfilename', type=str,
                        help="filename for checkpointed model to be stored.")
    parser.add_argument('-w', '--loadweights', type=str,
                        help='Loads weights from the checkpointed file (str).',
                        required=False)
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB()
    else:
        print("Option does not exist.")
        exit(0)

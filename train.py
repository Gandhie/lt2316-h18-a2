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


# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('train')
    ids = mycoco.query(args.categories, exclusive=True)
    #imgs = mycoco.iter_images(ids, args.categories)
    imgs, cats = mycoco.get_images_categories(ids, args.categories)
    #for id in ids:
        #print(len(id)) # test print
    #print(imgs[0])
    #print(cats[0])
    #print(len(imgs), len(cats))
    train_val_split = round(len(imgs) * 0.8)
    train_imgs = imgs[:train_val_split]
    val_imgs = imgs[train_val_split:]

    train_imgs_gen = mycoco.make_img_gen(train_imgs)
    val_imgs_gen = mycoco.make_img_gen(val_imgs)

    # input
    inputlayer = Input(shape=(200,200,3))
    # encoder
    conv2dlayer = Conv2D(8, (3,3))(inputlayer)
    relulayer = Activation('relu')(conv2dlayer)
    maxpool2dlayer = MaxPooling2D(pool_size=(2,2))(relulayer)
    conv2dlayer2 = Conv2D(16, (3,3))(maxpool2dlayer)
    relulayer2 = Activation('relu')(conv2dlayer2)
    maxpool2dlayer2 = MaxPooling2D(pool_size=(2,2))(relulayer2)
    encoded = maxpool2dlayer2
    # decoder
    upsamplinglayer = UpSampling2D((2,2))(maxpool2dlayer2)
    conv2dtranslayer = Conv2DTranspose(16, (3,3))(upsamplinglayer)
    relulayer3 = Activation('relu')(conv2dtranslayer)
    upsamplinglayer2 = UpSampling2D((2,2))(relulayer3)
    conv2dtranslayer2 = Conv2DTranspose(8, (3,3))(upsamplinglayer2)
    relulayer4 = Activation('relu')(conv2dtranslayer2)
    conv2dtranslayer3 = Conv2DTranspose(3, (3,3), activation='sigmoid')(relulayer4)
    # had issues with the iter_images generator data, since it has both image and category - resulting in wrong shape for the above layer. - Fixed by making lists (imgs and cats separate) and then generators from lists. But really bad results...?

    #load_weights here? For checkpointing.

    model = Model(inputlayer, conv2dtranslayer3)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # checkpoint
    filepath = '/scratch/gusastam/best.weights.h5' # change this to args.checkpointdir eventually. The arg should be almost the same as this line - add info about which model (we're saving 4 different ones).
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    steps = len(train_imgs) / 32 # setting batch size but probably not right. How to set batch size and where?
    valsteps = len(train_imgs) / 32

    model.fit_generator(train_imgs_gen, steps_per_epoch=steps, epochs=30, callbacks=callbacks_list, validation_data=(val_imgs_gen), validation_steps=valsteps)
    #loss: 0.5066 - acc: 0.0116 - val_loss: 0.5044 - val_acc: 0.0170 = seems really bad.

    #model.fit(train_imgs, train_imgs, epochs=3, callbacks=callbacks_list, validation_data=(val_imgs, val_imgs))
    """ ^
    Error when feeding fit(...): the list of images (before making generators again...)
    model.fit(train_imgs, train_imgs, epochs=3, callbacks=callbacks_list, validation_data=(val_imgs, val_imgs))

    ValueError: Error when checking model input: the list of Numpy arrays that you are passing to your model is not the size the model expected. Expected to see 1 array(s), but instead got the following list of 5158 arrays: [array([[[[0.87530637, 0.86354167, 0.84393382],
         [0.87132353, 0.8620098 , 0.84240196],
         [0.85943627, 0.85551471, 0.83590686],
         ...,
         [0.23768382, 0.21905637, 0.17689951...
    """

    print("Option A not fully implemented!")

    # What now?
    # Get access to the embeddings (see PIM from Asad and Demo 4 examples.)
    # Cluster the embeddings? How?
    # What is going in test.py?
    # Plot losses for all 4 models. How do we plot losses?
    # Output of compression layer for each training image, reduce dimensionality with PCA to 2D or 3D. Colour-code by category (min 3 cats), and plot the vectors. 

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

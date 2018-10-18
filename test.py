# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

# Add/update whatever imports you need.
from argparse import ArgumentParser
import mycoco
from keras.models import model_from_json
from keras import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D

# If you do option A, you may want to place your code here.  You can
# update the arguments as you need.
def optA():
    mycoco.setmode('test')
    if args.maxinstances != None:
        idsstart = mycoco.query(args.categories, exclusive=True)
        ids = []
        for id in idsstart:
            ids.append(list(id[:args.maxinstances]))
    else:
        ids = mycoco.query(args.categories, exclusive=True)

    # Gets image data (without labels).
    imgs = mycoco.iter_images_nocat(ids, args.categories, batch=32)
    # Gets image data (with labels).
    imgslabeled = mycoco.iter_images(ids, args.categories, batch=32)

    # Loading model from file.
    json_file = open(args.modelfile, "r")
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)

    # Compiling model.
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    #score = model.evaluate(imgs, imgs, verbose=0)
    #print(score)

    # Building second model for extracting embeddings.
    inputlayer = Input(shape=(200,200,3))
    # encoder
    conv2dlayer = Conv2D(8, (3,3), padding='same')(inputlayer)
    relulayer = Activation('relu')(conv2dlayer)
    maxpool2dlayer = MaxPooling2D(pool_size=(2,2))(relulayer)
    conv2dlayer2 = Conv2D(16, (3,3), padding='same')(maxpool2dlayer)
    relulayer2 = Activation('relu')(conv2dlayer2)
    maxpool2dlayer2 = MaxPooling2D(pool_size=(2,2))(relulayer2)
    encoded = maxpool2dlayer2

    # Creating second model.
    model2 = Model(inputlayer, maxpool2dlayer2)
    model2.summary()

    # Setting steps for predictions, as number of images divided by batch size.
    num_imgs = 0
    for id in ids:
        num_imgs += len(id)
    steps = round(num_imgs / 32)

    # Copying weights to a new model (only the encoder) and predicting to get embeddings.
    model2.set_weights(model.get_weights()[0:2])
    predictions = model2.predict_generator(imgslabeled, steps=steps)

    # Printing an example (since I am stopping here).
    print("An example prediction:")
    print(predictions[0])

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('test')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
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

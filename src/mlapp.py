
import utility

import os
import splitfolders
import glob
import json
import datetime
from timeit import default_timer as timer
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

g_model = None
g_classes = None

class TrainingCallback(Callback):
    def __init__(self, publish_func):
        super(TrainingCallback, self).__init__()
        self.publish_func = publish_func
    def on_epoch_begin(self, epoch, logs=None):
        self.starttime = timer()

def dummyCallback(epoch, epoch_time_sec):
    print("finished epoch", epoch, "epoch_time_sec", epoch_time_sec) 
    gpuinfo = json.dumps(utility.get_gpu_memory_usage())
    print("gpuinfo", gpuinfo)

def gpu_setup():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024*4.5)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

def model_setup(model_name):
    global g_model, g_classes
    g_model = tf.keras.models.load_model(os.path.join(utility.MODEL_PATH, model_name + '.h5'))
    g_classes = np.loadtxt(os.path.join(utility.MODEL_PATH, 'classes-' + model_name + '.txt'), dtype=str)

    return g_classes

def get_model_densenet121(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_densenet169(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_densenet201(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb0(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb1(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb3(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb4(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB4(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb5(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB5(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb6(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB6(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_efficientnetb7(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_inceptionresnetv2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_inceptionv3(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_mobilenet(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_mobilenetv2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_mobilenetv3large(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_mobilenetv3small(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.MobileNetV3Small(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_nasnetlarge(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.NASNetLarge(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_nasnetmobile(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.NASNetMobile(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet50(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet50v2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet50V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet101(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet101(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet101v2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet101V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers[:-15]:
        layer.trainable = False

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet152(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_resnet152v2(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_vgg16(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_vgg19(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def get_model_xception(numberOfClasses, img_height, img_width):
    model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    for layer in model.layers:
        layer.trainable = False

    x = model.output
    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    output  = Dense(units=numberOfClasses, activation='softmax')(x)
    return Model(model.input, output)

def training(train_model, classes, img_size=224, epochs=20, batch_size=8, learning_rate=0.01):
    img_height = img_size
    img_width = img_size

    utility.remove_my_dataset()

    # create the dataset train and val
    splitfolders.ratio(utility.DATA_PATH, output=utility.DATASET_PATH, seed=1773, ratio=(0.7, 0.2, 0.1), group_prefix=None)

    numberOfClasses = len(classes)
    trainingCallback = TrainingCallback(dummyCallback)
    


    train_data_gen = ImageDataGenerator(rotation_range=50,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1./255)
    valid_data_gen = ImageDataGenerator(rotation_range=45,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.3,
                                        horizontal_flip=True,
                                        vertical_flip=True,
                                        fill_mode='constant',
                                        cval=0,
                                        rescale=1./255)
    SEED = 1234
    tf.random.set_seed(SEED) 
    print("scan train directory")
    training_dir = os.path.join(utility.DATASET_PATH, 'train')
    train_gen = train_data_gen.flow_from_directory(training_dir,
                                                target_size=(img_height, img_width),
                                                batch_size=batch_size,
                                                classes=classes,
                                                class_mode='categorical',
                                                shuffle=True,
                                                seed=SEED) 

    print("scan val directory")
    valid_dir = os.path.join(utility.DATASET_PATH, 'val')
    valid_gen = valid_data_gen.flow_from_directory(valid_dir,
                                            target_size=(img_height, img_width),
                                            batch_size=batch_size, 
                                            classes=classes,
                                            class_mode='categorical',
                                            shuffle=False,
                                            seed=SEED)

    if train_model == utility.MODEL_DENSENET121:
        model = get_model_densenet121(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_DENSENET169:
        model = get_model_densenet169(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_DENSENET201:
        model = get_model_densenet201(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB0:
        model = get_model_efficientnetb0(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB1:
        model = get_model_efficientnetb1(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB2:
        model = get_model_efficientnetb2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB3:
        model = get_model_efficientnetb3(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB4:
        model = get_model_efficientnetb4(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB5:
        model = get_model_efficientnetb5(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB6:
        model = get_model_efficientnetb6(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_EFFICIENTNETB7:
        model = get_model_efficientnetb7(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_INCEPTIONRESNETV2:
        model = get_model_inceptionresnetv2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_INCEPTIONV3:
        model = get_model_inceptionv3(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_MOVBILENET:
        model = get_model_mobilenet(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_MOVBILENETV2:
        model = get_model_mobilenetv2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_MOVBILENETV3LARGE:
        model = get_model_mobilenetv3large(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_MOVBILENETV3SMALL:
        model = get_model_mobilenetv3small(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET50:
        model = get_model_resnet50(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET50V2:
        model = get_model_resnet50v2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET101:
        model = get_model_resnet101(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET101V2:
        model = get_model_resnet101v2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET152:
        model = get_model_resnet152(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_NASNETLARGE:
        model = get_model_nasnetlarge(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_NASNETMOBILE:
        model = get_model_nasnetmobile(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_RESNET152V2:
        model = get_model_resnet152v2(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_VGG16:
        model = get_model_vgg16(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_VGG19:
        model = get_model_vgg19(numberOfClasses, img_height, img_width)
    elif train_model == utility.MODEL_XCEPTION:
        model = get_model_xception(numberOfClasses, img_height, img_width)
    else:
        return
    
    # model.summary()
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])
    
    lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)

    st = EarlyStopping(monitor='loss',
                        min_delta=0.0,
                        patience=5,
                        verbose=1,
                        mode='min')

    callbacks = [lrr, st]

    # callbacks = [lrr]

    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    print("train_gen.n", train_gen.n, "train_gen.batch_size", train_gen.batch_size, "STEP_SIZE_TRAIN", STEP_SIZE_TRAIN)
    STEP_SIZE_VALID=valid_gen.n//valid_gen.batch_size
    print("valid_gen.n", valid_gen.n, "valid_gen.batch_size", valid_gen.batch_size, "STEP_SIZE_VALID", STEP_SIZE_VALID)
    
    transfer_learning_history = model.fit_generator(generator=train_gen,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_gen,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=[callbacks, trainingCallback],
    )
    
    model.save(os.path.join(utility.MODEL_PATH, train_model + '.h5'))
    np.savetxt(os.path.join(utility.MODEL_PATH, 'classes-' + train_model + '.txt'),  classes, fmt='%s')
    
    acc = transfer_learning_history.history['accuracy']
    val_acc = transfer_learning_history.history['val_accuracy']

    loss = transfer_learning_history.history['loss']
    val_loss = transfer_learning_history.history['val_loss']

    epochs_range = range(len(acc))

    now = datetime.datetime.now()
    file_timestamp = now.strftime('%Y%m%d_%H%M%S')

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(utility.RESULT_PATH, 'report-' + train_model + '_' + file_timestamp  + '.png'))

def inference(model_name, img_original, img_size=224):
    img0 = utility.resize(img_original, img_size)
    img = utility.remove_transparency(img0)

    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    if model_name == utility.MODEL_DENSENET121 or\
        model_name == utility.MODEL_DENSENET169  or\
        model_name == utility.MODEL_DENSENET201:
            x = tf.keras.applications.densenet.preprocess_input(x)
    elif model_name == utility.MODEL_EFFICIENTNETB0 or\
         model_name == utility.MODEL_EFFICIENTNETB1 or\
         model_name == utility.MODEL_EFFICIENTNETB2 or\
         model_name == utility.MODEL_EFFICIENTNETB3 or\
         model_name == utility.MODEL_EFFICIENTNETB4 or\
         model_name == utility.MODEL_EFFICIENTNETB5 or\
         model_name == utility.MODEL_EFFICIENTNETB6 or\
         model_name == utility.MODEL_EFFICIENTNETB7:
            x = tf.keras.applications.efficientnet.preprocess_input(x)
    elif model_name == utility.MODEL_INCEPTIONRESNETV2 or\
         model_name == utility.MODEL_INCEPTIONV3:
            x = tf.keras.applications.inception_v3.preprocess_input(x)
    elif model_name == utility.MODEL_MOVBILENET:
        x = tf.keras.applications.mobilenet.preprocess_input(x)
    elif model_name == utility.MODEL_MOVBILENETV2:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    elif model_name == utility.MODEL_MOVBILENETV3LARGE or\
         model_name == utility.MODEL_MOVBILENETV3SMALL:
            x = tf.keras.applications.mobilenet_v3.preprocess_input(x)
    elif model_name == utility.MODEL_NASNETLARGE or\
         model_name == utility.MODEL_NASNETMOBILE:
            x = tf.keras.applications.nasnet.preprocess_input(x)
    elif model_name == utility.MODEL_RESNET50:
        x = tf.keras.applications.resnet50.preprocess_input(x)
    elif model_name == utility.MODEL_RESNET101 or\
         model_name == utility.MODEL_RESNET152:
            x = tf.keras.applications.resnet.preprocess_input(x)
    elif model_name == utility.MODEL_RESNET50V2 or\
         model_name == utility.MODEL_RESNET101V2 or\
         model_name == utility.MODEL_RESNET152V2:
            x = tf.keras.applications.resnet_v2.preprocess_input(x)
    elif model_name == utility.MODEL_VGG16:
        x = tf.keras.applications.vgg16.preprocess_input(x)
    elif model_name == utility.MODEL_VGG19:
        x = tf.keras.applications.vgg19.preprocess_input(x)
    elif model_name == utility.MODEL_XCEPTION:
        x = tf.keras.applications.xception.preprocess_input(x)
    predictions = g_model.predict(x)
    score = tf.nn.softmax(predictions[0])
    # print("softmax predictions[0]", score)

    return score

def do_detect(model_name, img_original, img_size=224, num=3):
    score = inference(model_name, img_original, img_size)
    index_sort = np.argsort(-score)
    score_sort = np.sort(score)[::-1]

    result = result = '['

    for i in range(num):
        score = score_sort[i] * 100
        label = g_classes[index_sort[i]]
        result += '{"label":"' + label + '", "score":' + str(score) + '}, '

    result = result[0:-2] + ']'
    return result

if __name__ == '__main__':
    model = 'vgg19'
    gpu_setup()
    classes = utility.get_classes()
    training(model, classes, dummyCallback)

    model_setup(model)
    dirs = sorted(glob.glob(utility.TEST_DATA_PATH + '/*'))
    num = 0
    correct = 0

    for dir in dirs:
        dir_name = dir.replace(utility.TEST_DATA_PATH + '/', '')
        files = sorted(glob.glob(dir + '/*'))
        for file in files:
            image = Image.open(file)
            file_name = dir_name + '_' + file.replace(dir + '/', '')
            result = inference(model, image)
            result = '{"image":"' + file_name + '", "result":' + result + '}'
            num += 1
            jsonData = json.loads(result)
            if jsonData['result'][0]['label'] == dir_name:
                correct += 1
            else:
                print(result)

    print(str(correct) + ' / ' + str(num))
    utility.save_result(model, correct, num)
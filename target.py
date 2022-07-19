from dataLoader import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import metrics
from modelUtil import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)

DATA_NAME = "CIFAR"
MODEL = "ResNet50"
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
WEIGHTS_PATH = "weights/Target/{}_{}_{}.hdf5".format(DATA_NAME, MODEL, EPOCHS)
(x_train, y_train), (x_test, y_test), _= globals()['load_' + DATA_NAME]("Target")


def train(model, x_train, y_train):
    """
    Train the target model and save the weight of the model
    :param model: the model that will be trained
    :param x_train: the image as numpy format
    :param y_train: the label for x_train
    :param weights_path: path to save the model file
    :return: None
    """
    optim = keras.optimizers.Adam(lr=LEARNING_RATE)
    # optim = DPKerasSGDOptimizer(l2_norm_clip=1.0,
    #                           noise_multiplier=0.0001,
    #                           num_microbatches=1,
    #                           learning_rate=5e-2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optim,
                  metrics=[metrics.CategoricalAccuracy()])
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor='val_categorical_accuracy', verbose=1, save_best_only=True,
                                 mode='max')
    model.fit(x_train,
              y_train,
              validation_data=(x_test, y_test),
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              callbacks=[checkpoint])


def evaluate(x_test, y_test):
    model = keras.models.load_model(WEIGHTS_PATH)
    model.compile(loss="categorical"
                       "_crossentropy",
                  metrics=[metrics.CategoricalAccuracy(), metrics.Precision(), metrics.Recall()])
    loss, accuracy, precision, recall = model.evaluate(x_test, y_test, verbose=1)
    F1_Score = 2 * (precision * recall) / (precision + recall)
    print("loss:%.4f accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f"
          % (loss, accuracy, precision, recall, F1_Score))

TargetModel = globals()["create_{}_model".format(MODEL)](x_train.shape[1:], y_train.shape[1])
train(TargetModel, x_train, y_train)

evaluate(x_test, y_test)
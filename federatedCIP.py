import datetime
import numpy as np
import os
from tqdm import tqdm
from dataLoader import *
from modelUtil import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)

NUM_CLIENTS = 10
DATA_NAME = "CIFAR_Clients"
MODEL = "VGG16"
a = 0.3
EPOCHS = 1000
BATCH_SIZE = 64
LEARNING_RATE_gen = 1e-4


boundaries = [20000, 40000]
values = [0.001, 0.0005, 0.0001]
SAVED_EPOCHS = [200, 400, 500, 600, 700, 800, 900, 1000]
WEIGHTS_PATH_BASE_TEMPLATE = "weights/Gradient_Duel_{}/Generator/Client{}/{}_{}_{}_{}_{}.npy"
WEIGHTS_PATH_DISC_TEMPLATE = "weights/Gradient_Duel_{}/Discriminator/Client{}/{}_{}_{}_{}_{}.hdf5"
NOISE_PATH_TEMPLATE = "Base/Gradient_Duel_{}/Client{}/{}_{}_{}_{}_{}.png"

(x_train, y_train), (x_test, y_test), _ = globals()['load_' + DATA_NAME](NUM_CLIENTS)

train_datasets = [tf.data.Dataset.from_tensor_slices(
    (x, y)).batch(BATCH_SIZE) for x, y in zip(x_train, y_train)]
test_datasets = [tf.data.Dataset.from_tensor_slices(
    (x, y)).batch(BATCH_SIZE) for x, y in zip(x_test, y_test)]

def creat_duel_model(input_shape, output_shape):
    left_input = tf.keras.layers.Input(shape=input_shape, name='left_input')
    right_input = tf.keras.layers.Input(shape=input_shape, name='right_input')
    input_model = tf.keras.Sequential([ResNet50V2(include_top=False,
                 weights='imagenet',
                 input_shape=input_shape), GlobalAveragePooling2D()], name="model")
    concat = tf.keras.layers.Concatenate()([input_model(left_input), input_model(right_input)])
    logits = tf.keras.layers.Dense(output_shape)(concat)
    output = tf.keras.layers.Activation("softmax")(logits)
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=[output])
    model.summary()
    return model

models = [creat_duel_model((32, 32, 3), 100) for _ in range(NUM_CLIENTS)]

loss_fn = tf.keras.losses.CategoricalCrossentropy()
learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

generator_optimizer = tf.keras.optimizers.SGD(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate_fn)

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
train_acc_metric_agg = tf.keras.metrics.CategoricalAccuracy()

test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric_agg = tf.keras.metrics.CategoricalAccuracy()
test_alpha_acc_metric = tf.keras.metrics.CategoricalAccuracy()

overall_loss = tf.keras.metrics.CategoricalCrossentropy()

# bases = list(map(load_base_image, range(NUM_CLIENTS)))
bases = [gen_random_initial([32, 32, 3]) for i in range(NUM_CLIENTS)]

alpha = tf.constant(a, dtype=tf.float32)

@tf.function
def train_step(model, images, labels, base):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_tape.watch(base)
        inputs_A, inputs_B = blend_noise(images, base, alpha)
        outputs = model((inputs_A, inputs_B))
        ori_loss = loss_fn(labels, model((images, images), training=False))

        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(base))

        disc_loss = loss_1 - 1e-6 * ori_loss
        gen_loss = loss_1  + 1e-8 * loss_2
    gen_gradients = gen_tape.gradient(gen_loss, base)
    generator_optimizer.apply_gradients([(gen_gradients, base)])

    disc_gradients = disc_tape.gradient(disc_loss, model.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, model.trainable_variables))

    train_acc_metric.update_state(labels, outputs)
    overall_loss.update_state(labels, outputs)

def train():
    best_acc = 0
    for epoch in range(EPOCHS):
        for index in range(NUM_CLIENTS):
            with tqdm(enumerate(train_datasets[index])) as tBatch:
                for step, (images, labels) in tBatch:
                    train_step(models[index], images, labels, bases[index])
                    tBatch.set_description(
                        f"Model: {index} Epoch:{epoch + 1}, Step:{step}, Training Loss:{overall_loss.result()}, Training Acc:{train_acc_metric.result()}")

            with train_summary_writer.as_default():
                tf.summary.scalar(f'model {index} Acc', train_acc_metric.result(), step=epoch + 1)
            train_acc_metric.reset_states()
            overall_loss.reset_states()

        if epoch+1 in SAVED_EPOCHS:
            for i, model in enumerate(models):
                model.save(WEIGHTS_PATH_DISC_TEMPLATE.format(NUM_CLIENTS, i, DATA_NAME, MODEL, EPOCHS, epoch+1, a))
                np.save(WEIGHTS_PATH_BASE_TEMPLATE.format(NUM_CLIENTS, i, DATA_NAME, MODEL, EPOCHS, epoch+1, a), bases[i])
                tf.keras.preprocessing.image.save_img(NOISE_PATH_TEMPLATE.format(NUM_CLIENTS, i, DATA_NAME, MODEL, EPOCHS, epoch+1, a), bases[i][0])

        # Aggregate Part:
        print(f"Epoch {epoch+1} update weights")
        aggregate_weights(models)

        for index in range(NUM_CLIENTS):
            for images, labels in train_datasets[index]:
                images_train = blend_noise(images, bases[index], alpha)
                y_pred = models[index](images_train, training=False)

                train_acc_metric_agg.update_state(labels, y_pred)
        tf.print(f"Aggregated Training Acc: {train_acc_metric_agg.result()}")
        train_acc_metric_agg.reset_states()

        for index in range(NUM_CLIENTS):
            for images, labels in test_datasets[index]:
                images_test = blend_noise(images, bases[index], alpha)
                y_pred = models[index](images_test, training=False)
                test_acc_metric_agg.update_state(labels, y_pred)
        tf.print(f"Aggregated Testing Acc: {test_acc_metric_agg.result()}")
        test_acc_metric_agg.reset_states()
train()

import cv2 as cv
from modelUtil import *
from dataLoader import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)
DATA_NAME = "CIFAR"
MODEL = "ResNet50"
a=0.3
EPOCHS = 30
BASE = 10
BATCH_SIZE = 64
LEARNING_RATE_gen = 5e-5
LEARNING_RATE_disc = 1e-4
BASE_WEIGHTS = f'weights/Baseline/{DATA_NAME}_{MODEL}_{BASE}.hdf5'
WEIGHTS_PATH_BASE = f"weights/DPER/Generator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.npy"
WEIGHTS_PATH_DISC = f"weights/DPER/Discriminator/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.hdf5"
NOISE_PATH = f"Base/DPER/{DATA_NAME}_{MODEL}_{EPOCHS}_{a}_{BASE}.png"
(x_train, y_train), (x_test, y_test), _= globals()['load_' + DATA_NAME]("Target")

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(BATCH_SIZE)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)

def creat_dual_model(input_shape, output_shape, base_weights):
    base_model = tf.keras.models.load_model(base_weights)
    left_input = tf.keras.layers.Input(shape=input_shape, name='left_input')
    right_input = tf.keras.layers.Input(shape=input_shape, name='right_input')
    input_model = tf.keras.Sequential([ResNet50(include_top=False,
                 weights=None,
                 input_shape=input_shape), GlobalAveragePooling2D()], name="model")
    concat = tf.keras.layers.Concatenate()([input_model(left_input), input_model(right_input)])
    input_model.set_weights(base_model.layers[0].get_weights())
    logits = tf.keras.layers.Dense(output_shape)(concat)
    output = tf.keras.layers.Activation("softmax")(logits)
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=[output])
    model.layers[-2].set_weights([np.tile(base_model.layers[-2].get_weights()[0], (2, 1)), base_model.layers[-2].get_weights()[1]])
    model.summary()
    return model

baseline = creat_dual_model(x_train.shape[1:], y_train.shape[1], BASE_WEIGHTS)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_disc)
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
test_acc_metric = tf.keras.metrics.CategoricalAccuracy()
overall_loss = tf.keras.metrics.CategoricalCrossentropy()
noise = tf.random.uniform([32, 32, 3], minval=0, maxval=1)
alpha = tf.constant(a, dtype=tf.float32)
base = tf.Variable(noise[tf.newaxis, :], trainable=True)


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_tape.watch(base)
        inputs_A, inputs_B = blend_noise(images, base, alpha)
        outputs = baseline((inputs_A, inputs_B))
        ori_loss = loss_fn(labels, baseline((images, images), training=False))

        loss_1 = loss_fn(labels, outputs)
        loss_2 = tf.reduce_sum(tf.abs(base))
        disc_loss = loss_1 - 1e-12 * ori_loss
        gen_loss = loss_1 + 1e-8 * loss_2

    gen_gradients = gen_tape.gradient(gen_loss, base)
    generator_optimizer.apply_gradients([(gen_gradients, base)])
    disc_gradients = disc_tape.gradient(disc_loss, baseline.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(disc_gradients, baseline.trainable_variables))
    train_acc_metric.update_state(labels, outputs)
    overall_loss.update_state(labels, outputs)


test_acc = 0
for epoch in range(EPOCHS):
    with tqdm(enumerate(train_dataset)) as tBatch:
        for step, (images, labels) in tBatch:
            train_step(images, labels)
            tBatch.set_description("Epoch:{}, Step:{}, Training Loss:{}, Training Acc:{}"
                                    .format(epoch + 1, step, overall_loss.result(), train_acc_metric.result()))
        train_acc_metric.reset_states()
        overall_loss.reset_states()
    for images, labels in test_dataset:
        generated_image = base
        images_test = blend_noise(images, generated_image, alpha)
        y_pred = baseline(images_test, training=False)
        test_acc_metric.update_state(labels, y_pred)
    tf.print(f"Acc with noise: {test_acc_metric.result()}")
    test_acc_metric.reset_states()

# This is a sample Python script.
# VGG16 is implemented in tensorflow
# pay attention to axis=-1
# for GPU, axis should be equal to 1
import tensorflow as tf

class Vgg16(tf.keras.Model):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.max_pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.max_pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.max_pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')
        self.max_pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2], padding='same')

        self.conv_layer64 = tf.keras.layers.Conv2D(filters=64,kernel_size =3,  padding='same',
                                                   data_format="channels_last")
        self.conv_layer64_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                                                   data_format="channels_last")


        self.conv_layer128 = tf.keras.layers.Conv2D(filters=128,kernel_size =3,  padding='same',
                                                    data_format="channels_last")
        self.conv_layer128_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                                                    data_format="channels_last")


        self.conv_layer256 = tf.keras.layers.Conv2D(filters=256, kernel_size =3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer256_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer256_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                                                    data_format="channels_last")


        self.conv_layer512 = tf.keras.layers.Conv2D(filters=512, kernel_size =3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer512_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer512_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer512_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer512_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                                    data_format="channels_last")
        self.conv_layer512_6 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same',
                                                      data_format="channels_last")

        self.batch_normalization = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization2 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization3 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization4 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization5 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization6 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization7 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization8 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization9 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization10 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization11 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization12 = tf.keras.layers.BatchNormalization(axis=-1)
        self.batch_normalization13 = tf.keras.layers.BatchNormalization(axis=-1)

        self.relu_activation = tf.keras.layers.Activation('relu')
        self.relu_activation2 = tf.keras.layers.Activation('relu')

    def call(self, inputs,units):
        conv1_1 = self.conv_layer64(inputs)
        conv1_1_b = self.batch_normalization(conv1_1,training = True)
        conv1_1_b = tf.keras.layers.Activation('relu')(conv1_1_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization.updates)
        conv1_2 = self.conv_layer64_2(conv1_1_b)
        conv1_2_b = self.batch_normalization2(conv1_2,training = True)
        conv1_2_b = tf.keras.layers.Activation('relu')(conv1_2_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization2.updates)

        pool1 = self.max_pool(conv1_2_b)

        conv2_1 = self.conv_layer128(pool1)
        # conv2_1 = tf.nn.relu(conv2_1) #tf.nn.relu
        conv2_1_b = self.batch_normalization3(conv2_1,training = True)
        conv2_1_b = tf.keras.layers.Activation('relu')(conv2_1_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization3.updates)
        conv2_2 = self.conv_layer128_2(conv2_1_b)
        conv2_2_b = self.batch_normalization4(conv2_2,training = True)
        conv2_2_b = tf.keras.layers.Activation('relu')(conv2_2_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization4.updates)

        pool2 = self.max_pool2(conv2_2_b)

        conv3_1 = self.conv_layer256(pool2)
        conv3_1_b = self.batch_normalization5(conv3_1,training = True)
        conv3_1_b = tf.keras.layers.Activation('relu')(conv3_1_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization5.updates)

        conv3_2 = self.conv_layer256_2(conv3_1_b)
        conv3_2_b = self.batch_normalization6(conv3_2,training = True)
        conv3_2_b = tf.keras.layers.Activation('relu')(conv3_2_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization6.updates)

        conv3_3 = self.conv_layer256_3(conv3_2_b)
        conv3_3_b = self.batch_normalization7(conv3_3,training = True)
        conv3_3_b = tf.keras.layers.Activation('relu')(conv3_3_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization7.updates)

        pool3 = self.max_pool3(conv3_3_b)

        conv4_1 = self.conv_layer512(pool3)
        conv4_1_b = self.batch_normalization8(conv4_1,training = True)
        conv4_1_b = tf.keras.layers.Activation('relu')(conv4_1_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization8.updates)

        conv4_2 = self.conv_layer512_2(conv4_1_b)
        conv4_2_b = self.batch_normalization9(conv4_2,training = True)
        conv4_2_b = tf.keras.layers.Activation('relu')(conv4_2_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization9.updates)

        conv4_3 = self.conv_layer512_3(conv4_2_b)
        conv4_3_b = self.batch_normalization10(conv4_3,training = True)
        conv4_3_b = tf.keras.layers.Activation('relu')(conv4_3_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization10.updates)

        pool4 = self.max_pool4(conv4_3_b)

        conv5_1 = self.conv_layer512_4(pool4)
        conv5_1_b = self.batch_normalization11(conv5_1,training = True)
        conv5_1_b = tf.keras.layers.Activation('relu')(conv5_1_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization11.updates)

        conv5_2 = self.conv_layer512_5(conv5_1_b)
        conv5_2_b = self.batch_normalization12(conv5_2,training = True)
        conv5_2_b = tf.keras.layers.Activation('relu')(conv5_2_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization12.updates)

        conv5_3 = self.conv_layer512_6(conv5_2_b)
        conv5_3_b = self.batch_normalization13(conv5_3,training = True)
        conv5_3_b = tf.keras.layers.Activation('relu')(conv5_3_b)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, self.batch_normalization13.updates)

        pool5 = self.max_pool5(conv5_3_b)

        fc6 = tf.keras.layers.Flatten()(pool5)
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        relu6 = self.relu_activation(fc6)

        fc7 = tf.keras.layers.Dense(units=4096, activation='relu')(relu6)
        relu7 = self.relu_activation2(fc7)

        fc8 = tf.keras.layers.Dense(units=4096, activation='relu')(relu7)

        outputs = tf.keras.layers.Dense(units=units, activation='softmax')(fc8) #3?

        return outputs

    def get_model(self,inputs,units):
        out_puts = self.call(inputs,units)
        return tf.keras.models.Model(inputs=inputs, outputs=out_puts)

def get_vgg16_model():
    tf.keras.backend.set_learning_phase(True)
    vgg16_model = Vgg16()
    # inputs = tf.keras.Input(shape=(224, 224, 3),dtype=tf.float32)
    vgg16_model = vgg16_model.get_model(tf.keras.Input(shape=(224,224,3)),5)
    print(vgg16_model.summary())
    adamoptimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    vgg16_model.compile(optimizer=adamoptimizer, loss="sparse_categorical_crossentropy",
                  metrics=["sparse_categorical_accuracy"])
    return True
if __name__ == '__main__':
    # print("VGG16")
    # inputs = tf.keras.layers.Input(shape=(3,))
    get_vgg16_model()
    # model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
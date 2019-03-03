import tensorflow as tf


def ResidualBlock(input):
    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=input,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1
    )

    # Batch normalization #1
    batchnorm1 = tf.layers.batch_normalization(conv1)

    # ReLU #1
    relu1 = tf.nn.relu(batchnorm1)

    # Convolutional layer #2
    conv2 = tf.layers.conv2d(
        inputs=relu1,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1
    )

    # Batch normalization #2
    batchnorm2 = tf.layers.batch_normalization(conv2)

    # Skip connection
    skip = tf.add(batchnorm2, input)

    # ReLU #2
    relu2 = tf.nn.relu(skip)

    return relu2


def ResidualTower(input, n_blocks):
    res = input
    for _ in range(n_blocks):
        res = ResidualBlock(res)
    return res


def AlphaGo19Net(inputs, labels, n_res_blocks, learning_rate):
    # Convolutional layer #1
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=236,
        kernel_size=[4, 4],
        padding='same',
        strides=1
    )

    # Batch normalization layer #1
    batchnorm1 = tf.layers.batch_normalization(conv1)

    # ReLU layer #1
    relu2 = tf.nn.relu(batchnorm1)

    # Tower of residual blocks
    tower = ResidualTower(relu2, n_res_blocks)

    # Convolutional layer #3
    conv3 = tf.layers.conv2d(
        inputs=tower,
        filters=2,
        kernel_size=[1, 1],
        padding='same',
        strides=1
    )

    # Batch normalization layer #4
    batchnorm4 = tf.layers.batch_normalization(conv3)

    # ReLU layer #4
    relu4 = tf.nn.relu(batchnorm4)

    # Fully connected layer
    with tf.name_scope('Predicted'):
        relu4 = tf.reshape(relu4, [-1, 6 * 7 * 2])
        pred = tf.layers.dense(inputs=relu4, units=7)

    # Loss
    with tf.name_scope('Loss'):
        loss = tf.losses.mean_squared_error(
            labels=labels, predictions=pred)

    # Configure optimizer
    with tf.name_scope('Adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # Accuracy
    with tf.name_scope('Accuracy'):
        acc = tf.equal(pred, labels)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))

    # train_op = optimizer.minimize(
    #     loss=loss,
    #     global_step=tf.train.global_step()
    # )
    return pred, loss, optimizer, acc

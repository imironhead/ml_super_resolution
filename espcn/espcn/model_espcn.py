"""
"""
import tensorflow as tf


def build_model(lr_source, scaling_factor=3, hr_target=None):
    """
    build espcn model

    lr_source:  source image batch tensor to be super resolved
    hr_target:  target image batch as training labels in sub-pixel convolved
                shape, build a partial model for testing if hr_target is None
    scaling_factor:
                factor of up scaling. the delpth of the final layer should be
                3 * scaling_factor * scaling_factor (sub-pixel layer)
    """
    model = {
        'lr_source': lr_source,
    }

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: arXiv:1609.05158v2, 3.2
    #       for the espcn, we set l = 3, (f1, n1) = (5, 64), (f2, n2) = (3, 32)
    #       and f3 = 3 in our evaluations.
    #
    # NOTE: arXiv:1609.05158v2, 3.2
    #       we choose tanh instead ofrelu as the activation function for the
    #       final model motivated by our experimental results.
    tensors = tf.layers.conv2d(
        lr_source,
        filters=64,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        name='f1')

    tensors = tf.layers.conv2d(
        tensors,
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        name='f2')

    # NOTE: arXiv:1609.05158v2, 2.2
    #       note that we do not apply nonlinearity to the outputs of the
    #       convolution at the last layer.
    # NOTE: fit depth to sub-pixel convolution layer
    sr_result = tf.layers.conv2d(
        tensors,
        filters=3 * (scaling_factor ** 2),
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        name='f3')

    model['sr_result'] = sr_result

    if hr_target is None:
        # NOTE: we do not want to train this model (freeze it to gain better
        #       performance)
        return model

    model['hr_target'] = hr_target

    # NOTE: arXiv:1609.05158v2, 2.2
    #       and calculate the pixel-wise mean squared error (MSE) of the
    #       reconstruction as an objective function to train the network.
    loss = tf.losses.mean_squared_error(
        sr_result, hr_target, reduction=tf.losses.Reduction.MEAN)

    # NOTE: arXiv:1609.05158v2, 3.2
    #       initial learning rate is  set to 0.01 and final learning rate is
    #       set to 0.0001 and updated gradually when the improvement of the
    #       cost function is smaller than a threshold mu.
    learning_rate = tf.placeholder(shape=[], dtype=tf.float32)

    step = tf.train.get_or_create_global_step()

    optimizer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    model['step'] = step
    model['loss'] = loss
    model['optimizer'] = optimizer
    model['learning_rate'] = learning_rate

    return model


def build_test_model(meta_path, ckpt_path):
    """
    use extract_weights to extract trained weights from a checkpoint. then use
    the weights to build a test model without variables (light weight & dynamic
    image size).
    """
    variables = extract_weights(meta_path, ckpt_path)

    # NOTE: get the scaling factor
    scaling_factor = int((variables['f3/bias:0'].size // 3) ** 0.5)

    # NOTE: arrays to constant tensors
    # NOTE: f3/bias:0 is not a valid name (can not have ':'?)
    variables = {k: tf.constant(v, name=k[:-2]) for k, v in variables.items()}

    lr_sources = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32)

    # NOTE: 1st layer
    tensors = tf.nn.conv2d(
        lr_sources, variables['f1/kernel:0'], [1] * 4, padding='SAME')
    tensors = tf.nn.bias_add(tensors, variables['f1/bias:0'])
    tensors = tf.nn.tanh(tensors)

    # NOTE: 2nd layer
    tensors = tf.nn.conv2d(
        tensors, variables['f2/kernel:0'], [1] * 4, padding='SAME')
    tensors = tf.nn.bias_add(tensors, variables['f2/bias:0'])
    tensors = tf.nn.tanh(tensors)

    # NOTE: 3rd layer
    # NOTE: arXiv:1609.05158v2, 2.2
    #       note that we do not apply nonlinearity to the outputs of the
    #       convolution at the last layer.
    tensors = tf.nn.conv2d(
        tensors, variables['f3/kernel:0'], [1] * 4, padding='SAME')
    tensors = tf.nn.bias_add(tensors, variables['f3/bias:0'])

    sr_results = tensors

    # NOTE: we can not build the super-resolved result here!
    #       because the size of input images is unknown so we can not know
    #       how to split the sub-pixel results here (split across the
    #       horizontal axis with a constant witch is width of the input image).

    return {
        'lr_sources': lr_sources,
        'sr_results': sr_results,
        'scaling_factor': scaling_factor,
    }


def extract_weights(meta_path, ckpt_path):
    """
    """
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(meta_path)

        saver.restore(session, ckpt_path)

        # NOTE: collect all kernels & bias
        variables = tf.trainable_variables()

        variables = {v.name: session.run(v) for v in variables}

    # NOTE: we no longer need the training model.
    tf.reset_default_graph()

    return variables


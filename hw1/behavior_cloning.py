import pickle
import tensorflow as tf
import numpy as np
import os
import argparse


# Train an agent using behavior cloning:
#   e.g. python behavior_cloning.py Hopper-v2

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()


def generate_training_data(data):
    inputs = data['observations']
    outputs = np.squeeze(data['actions'], axis=1)
    input_dim = inputs.shape[-1]
    output_dim = outputs.shape[-1]
    return inputs, outputs, input_dim, output_dim


def create_model(input_dim, output_dim):
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None, output_dim])

    layer = tf.layers.dense(input_ph, 64,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                            name='hidden_layer_1'
                            )
    layer = tf.layers.dense(layer, 128,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                            name='hidden_layer_2'
                            )
    layer = tf.layers.dense(layer, 64,
                            activation=tf.nn.relu,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                            name='hidden_layer_3'
                            )
    layer = tf.layers.dense(layer, output_dim,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                            name='output_layer'
                            )
    output_pred = layer
    return input_ph, output_ph, output_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str, default='Hopper-v2')
    args = parser.parse_args()
    envname = args.envname

    batch_size = 128
    learning_rate = .001
    regularization = 1e-6
    max_steps = 50000

    train_dir = '/tmp/cs294_hw1/behavior_cloning'
    expert_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'expert_data/')

    checkpoint_dir = os.path.join(train_dir, envname)
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, 'model.ckpt')

    expert_data_filename = os.path.join(expert_data_dir, envname + '.pkl')
    with open(expert_data_filename, 'rb') as file:
        data = pickle.load(file)
    inputs, outputs, input_dim, output_dim = generate_training_data(data)

    sess = tf_reset()

    input_ph, output_ph, output_pred = create_model(input_dim, output_dim)

    loss = tf.losses.mean_squared_error(output_ph, output_pred)
    loss += regularization * tf.losses.get_regularization_loss()
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for training_step in range(max_steps):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=len(inputs), size=batch_size)
        input_batch = inputs[indices]
        output_batch = outputs[indices]

        _, loss_run = sess.run([opt, loss], feed_dict={input_ph: input_batch, output_ph: output_batch})

        if training_step % 1000 == 0:
            print('{0:04d} loss: {1:.3g}'.format(training_step, loss_run))
            saver.save(sess, model_path)

    return


if __name__ == '__main__':
    main()

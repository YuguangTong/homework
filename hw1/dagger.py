import pickle
import tensorflow as tf
import numpy as np
import os
import argparse
import gym
import load_policy


class Config:
    def __init__(self,
                 input_dim=10,
                 output_dim=10,
                 batch_size=128,
                 learning_rate=5e-4,
                 regularization=1e-6,
                 num_epochs=20
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.num_epochs = num_epochs


class Agent:
    def __init__(self, config):
        self.config = config
        self.add_placeholder()
        self.add_layers()
        self.add_loss()
        self.add_optimizer()

    def add_placeholder(self):
        self.input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.config.input_dim])
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self.config.output_dim])

    def add_layers(self):
        layer = tf.layers.dense(self.input_placeholder, 64,
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
        layer = tf.layers.dense(layer, self.config.output_dim,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.),
                                name='output_layer'
                                )
        self.output_operator = layer

    def add_loss(self):
        self.loss_operator = tf.losses.mean_squared_error(self.output_placeholder, self.output_operator)
        self.loss_operator += self.config.regularization * tf.losses.get_regularization_loss()

    def add_optimizer(self):
        self.optimizer_operator = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(
            self.loss_operator)

    def train_on_batch(self, sess, x, y):
        loss, _ = sess.run([self.loss_operator, self.optimizer_operator],
                           feed_dict={self.input_placeholder: x, self.output_placeholder: y})
        return loss

    def predict(self, sess, x):
        pred = sess.run(self.output_operator, feed_dict={self.input_placeholder: x})
        return pred


def run_model(sess, env, model):
    max_steps = env.spec.timestep_limit
    observations = []
    obs = env.reset()
    done = False
    totalr = 0.
    steps = 0
    while not done and steps < max_steps:
        action = model.predict(sess, obs[None, :])
        obs, r, done, _ = env.step(action)
        observations.append(obs)
        totalr += r
        steps += 1
    return totalr, observations


def expand_data(sess, env, model, expert_policy, num_runs):
    new_obs = []
    new_actions = []
    for _ in range(num_runs):
        _, obs = run_model(sess, env, model)
        new_obs.extend(obs)
        actions = [expert_policy(x[None, :])[0] for x in obs]
        new_actions.extend(actions)
    return new_obs, new_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str, default='Hopper-v2')
    args = parser.parse_args()
    print('train dagger on {}'.format(args.envname))

    # model checkpoint directory
    train_dir = '/tmp/cs294_hw1/dagger'
    checkpoint_dir = os.path.join(train_dir, args.envname)
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, 'model.ckpt')

    # expert policy
    expert_policy_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experts/' + args.envname + '.pkl')

    # load initial training data
    expert_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'expert_data/')
    expert_data_filename = os.path.join(expert_data_dir, args.envname + '.pkl')
    with open(expert_data_filename, 'rb') as file:
        data = pickle.load(file)
    X = data['observations']
    y = np.squeeze(data['actions'], axis=1)
    input_dim = X.shape[-1]
    output_dim = y.shape[-1]

    # model config
    config = Config(input_dim=input_dim, output_dim=output_dim)

    tf.reset_default_graph()
    model = Agent(config)
    expert_policy = load_policy.load_policy(expert_policy_file)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=5, )

        # repeat: train model + expand data
        env = gym.make(args.envname)

        num_rollouts = 10
        num_expansion = 15

        rewards = []

        for i in range(num_expansion):
            # train model
            max_steps = int(config.num_epochs * len(X) / config.batch_size)
            for training_step in range(max_steps):
                # get a random subset of the training data
                indices = np.random.randint(low=0, high=len(X), size=config.batch_size)
                X_batch = X[indices]
                y_batch = y[indices]
                loss = model.train_on_batch(sess, X_batch, y_batch)

                if training_step % 1000 == 0:
                    print('{0:04d} loss: {1:.3g}'.format(training_step, loss))
                    saver.save(sess, model_path)

            # generate new data
            new_X, new_y = expand_data(sess, env, model, expert_policy, num_rollouts)

            print('-' * 50)
            print('add {0} samples in iteration {1}'.format(len(new_X), i))
            print('-' * 50)

            X = np.concatenate([X, np.array(new_X)])
            y = np.concatenate([y, np.array(new_y)])

            # evaluate performance
            reward = []
            for _ in range(15):
                r, _ = run_model(sess, env, model)
                reward.append(r)
            rewards.append(reward)
            print('mean reward in iteration {0} is {1:.3g}'.format(i, np.mean(reward)))
    stats_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  'result/' + args.envname + '_dagger_learning_curve.npy')
    np.save(stats_filename, np.array(rewards))


if __name__ == '__main__':
    main()

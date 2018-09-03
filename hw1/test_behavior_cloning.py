import argparse
from behavior_cloning import create_model, tf_reset
import gym
import numpy as np
import os
import tensorflow as tf

# Run agent trained by behavior cloning and collect stats:
#   e.g. python test_behavior_cloning.py Hopper-v2 --num_rollouts 5 --render

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str, default='Hopper-v2')
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    sess = tf_reset()
    env = gym.make(args.envname)
    output_dim = env.action_space.shape[0]
    input_dim = env.observation_space.shape[0]
    train_dir = '/tmp/cs294_hw1/behavior_cloning'
    checkpoint_dir = os.path.join(train_dir, args.envname)
    model_path = os.path.join(checkpoint_dir, 'model.ckpt')

    input_ph, output_ph, output_pred = create_model(input_dim, output_dim)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = sess.run([output_pred], feed_dict={input_ph: obs[None, :]})[0]
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))


if __name__ == '__main__':
    main()

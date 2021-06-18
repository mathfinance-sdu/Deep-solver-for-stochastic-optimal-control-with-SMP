import time
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')

class Solver(object):

    def __init__(self,):
        self.valid_size = 256
        self.batch_size = 64
        self.num_iterations = 4000
        self.print_frequency = 200
        self.lr_values = [5e-3, 1e-3, 5e-4]
        self.lr_boundaries = [5000, 8000]
        self.config = Config()

        self.model = WholeNet()
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(self.lr_boundaries, self.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []
        dW = self.config.sample(self.valid_size)
        valid_data = dW
        for step in range(self.num_iterations+1):
            if step % self.print_frequency == 0:
                cost = self.model(valid_data, training=True)
                elapsed_time = time.time() - start_time
                training_history.append([step, cost, elapsed_time])
                print("step: %5u, Cost: %.4e, elapsed time: %3u" % (step, cost, elapsed_time))
            self.train_step(self.config.sample(self.batch_size))
        self.training_history = training_history

    @tf.function
    def train_step(self, train_data):
        """Updating the gradients"""
        with tf.GradientTape(persistent=True) as tape:
            loss = self.model(train_data, training = True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))

class WholeNet(tf.keras.Model):
    """Building the neural network architecture"""
    def __init__(self):
        super(WholeNet, self).__init__()
        self.config = Config()
        self.subnet = [FNNet() for _ in range(self.config.num_time_interval)]

    def call(self, dw, training):
        x_init = tf.ones([1, self.config.dim_x], dtype=tf.dtypes.float64) * 1.0
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t
        all_ones = tf.ones([tf.shape(dw)[0], 1], dtype=tf.dtypes.float64)

        # initial state
        x = tf.matmul(all_ones, x_init)

        # initial cost functional
        l = 0.0 
        for t in range(0, self.config.num_time_interval):
            u = self.subnet[t](x, training)

            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t
            x = x + self.config.b_fn(time_stamp[t], x, u) * self.config.delta_t + self.config.sigma_fn(time_stamp[t], x, u) * dw[:, :, t]

        # terminal condition
        l = l + self.config.h_fn(self.config.total_time, x)
        cost = tf.reduce_mean(l) 

        return cost

class FNNet(tf.keras.Model):
    def __init__(self):
        super(FNNet, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x+10, self.config.dim_x+10, self.config.dim_x+10]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim_u
        self.dense_layers.append(tf.keras.layers.Dense(self.config.dim_u, activation=None))

    def call(self, x, training):
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training)
        return x

class Config(object):
    """function config"""
    def __init__(self):
        super(Config, self).__init__()
        self.dim_x = 100
        self.dim_u = 100
        self.num_time_interval = 50
        self.total_time = 0.1
        self.delta_t = (self.total_time + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    # sample function
    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.dim_u, self.num_time_interval]) * self.sqrth
        return dw_sample

    # cost functional integral item
    def f_fn(self, t, x, u):
        return tf.reduce_sum(u ** 2, 1, keepdims=True)

    # cost fuctional terminal function
    def h_fn(self, t, x):
        return 0.5 * tf.reduce_sum(x ** 2, 1, keepdims=True)

    def b_fn(self, t, x, u):
        return tf.sin(u)

    def sigma_fn(self, t, x, u):
        return x

def main():
    solver = Solver()
    print('Training time %3u:' % (1))
    solver.train()
    data = np.array(solver.training_history)
    k = 10
    output = np.zeros((len(data[:, 0]), 2 + k))
    output[:, 0] = data[:, 0]
    output[:, 1] = data[:, 2]
    output[:, 2] = data[:, 1]
    for i in range(k - 1):
        print('Training time %3u:' % (i + 2))
        solver = Solver()
        solver.train()
        data = np.array(solver.training_history)
        output[:, 3 + i] = data[:, 1]

    a = ['%d', '%.5e']
    for _ in range(k):
        a.append('%.5e')
    np.savetxt("./Data_d100_control.csv", output, fmt=a, delimiter=',')

    print('Solving is done!')

if __name__ == '__main__':
    main()
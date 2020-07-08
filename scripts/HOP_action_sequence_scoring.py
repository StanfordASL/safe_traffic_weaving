#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import timeit
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import rospy
import numpy as np
import tensorflow as tf
from safe_traffic_weaving.msg import PredictionInput, PredictionOutput
from utils.numpy_ros import numpy_to_multiarray, multiarray_to_numpy
dir(tf.contrib)

model_dir = rospy.get_param("model", "") or os.path.join(os.path.dirname(__file__), "../models/traffic_weaving_prediction")
viz_sample_count = rospy.get_param("human_viz_sample_count", 50)
time_scale_factor = rospy.get_param("time_scale_factor", 1)
x_scale_factor = rospy.get_param("x_scale_factor", 1)
print("Loading model from: " + model_dir)
print("Collision avoidance x_scale_factor: " + str(x_scale_factor))

def unstandardize(tensor, mean, std, include_bias=True):
    tile_ct = int(tensor.shape[-1].value / mean.shape[-1].value)
    if include_bias:
        return tensor * tf.tile(std, [tile_ct]) + tf.tile(mean, [tile_ct])
    else:
        return tensor * tf.tile(std, [tile_ct])

def score_trajectories(y, car1, car2, traj_length, car1_future):
    k, bs, _, _ = tf.unstack(tf.shape(y))
    ph = y.shape[2].value
    # with tf.variable_scope("scoring"):
    with tf.compat.v1.variable_scope("scoring"):
        curr_step = traj_length - 1
        car2_present = tf.expand_dims(tf.expand_dims(car2[:,curr_step,:], 0), 0)        # [1, 1, 1, 6]
        car2_present_p, car2_present_v, car2_present_a = tf.split(car2_present, 3, axis=3)                     # [1, 1, 1, 2]x3

        car2_future_a = y                                                                                      # [k, bs, ph, 2]
        car2_present_and_future_a = tf.concat([tf.tile(car2_present_a, [k, bs, 1, 1]), car2_future_a], axis=2) # [k, bs, ph+1, 2]
        car2_future_v = car2_present_v + tf.cumsum((car2_present_and_future_a[:,:,:-1,:] +
                                                    car2_present_and_future_a[:,:,1:,:])/2, axis=2)/10         # [k, bs, ph, 2]
        car2_present_and_future_v = tf.concat([tf.tile(car2_present_v, [k, bs, 1, 1]), car2_future_v], axis=2) # [k, bs, ph+1, 2]
        car2_future_p = car2_present_p + tf.cumsum((car2_present_and_future_v[:,:,:-1,:] +
                                                    car2_present_and_future_v[:,:,1:,:])/2, axis=2)/10         # [k, bs, ph, 2]

        car1_future_p, car1_future_v, car1_future_a = tf.split(car1_future, 3, axis=2)    # [bs, ph, 2]x3
        delta_p = car2_future_p - car1_future_p                                           # [k, bs, ph, 2]

        # proximity penalty
        dx_danger, dy_danger = 8/x_scale_factor, 2
        r_danger = np.hypot(dx_danger, dy_danger)
        cars_close = tf.reduce_all(tf.abs(delta_p) < [dx_danger, dy_danger], axis=-1)              # [k, bs, ph]
        prox_penalty = -1000*tf.cast(cars_close, tf.float32)*(1 + r_danger - tf.norm(delta_p, axis=-1))    # [k, bs, ph]

        # time of crossover reward (maybe delta_x should be adjusted for car length)
        delta_v = car2_future_v - car1_future_v            # [k, bs, ph, 2]
        delta_x = delta_p[:,:,:,0]
        delta_xd = delta_v[:,:,:,0]
        ntco = tf.sign(delta_x)*tf.maximum(tf.abs(delta_x) - 3, 0)*delta_xd # negative crossover time
        ntco = tf.maximum(ntco, 0)    # ignore positive crossover time, we just want to reward negative
        tco_reward = 100*tf.minimum(ntco, 1)    # [k, bs, ph]

        # car 2 extreme speed penalty (penalty for cutting off; -20 for each m/s below 24 m/s)
        car2_speed_penalty = 50*tf.minimum(car2_future_v[:,:,:,0] - 24, 0)    # [k, bs, ph]

        # car 1 extreme speed penalty
        car1_speed_penalty = (10*tf.minimum(car1_future_v[:,:,0] - 24, 0) -
                              10*tf.maximum(car1_future_v[:,:,0] - 36, 0))    # [bs, ph]

        # car 1 goal penalty
        # y_top, y_bot = -1.83, -6.09
        y_top, y_bot = -1.5, -6.5
        y_mid = (y_top + y_bot)/2
        top_start_indicator = tf.to_float(car1[0,0,1] > y_mid)
        bot_start_indicator = 1 - top_start_indicator
        urgency = 500*tf.minimum(1.2 + car1_future_p[:,:,0]/150 + .6, 1)
        car1_goal_penalty = -(top_start_indicator*urgency*tf.minimum(tf.abs(car1_future_p[:,:,1] - y_bot), 2) +
                              bot_start_indicator*urgency*tf.minimum(tf.abs(car1_future_p[:,:,1] - y_top), 2))    # [bs, ph]

        # car 1 acceleration penalty
        car1_accel_penalty = -tf.square(car1_future_a[:,:,0]/5)    # *car1_future_a[:,:,0]/10
        car1_accel_tv_penalty = -tf.reduce_sum(tf.abs(car1_future_a[:,1:,0] - car1_future_a[:,:-1,0]), -1)/20     # [bs]

        rewards = (prox_penalty + tco_reward + car2_speed_penalty +
                   car1_speed_penalty + car1_goal_penalty + car1_accel_penalty)                                   # [k, bs, ph]
        tree = tf.reduce_sum(rewards*np.exp(np.log(0.9)*np.arange(ph)), -1) + car1_accel_tv_penalty               # [k, bs]

        tree = tf.reshape(tree, [-1, 8, 512])
        a1 = tf.argmax(tf.reduce_mean(tf.reduce_max(tree, -1), 0), 0)
        tree = tf.reshape(tree[:,a1,:], [-1, 8, 64])
        a2 = tf.argmax(tf.reduce_mean(tf.reduce_max(tree, -1), 0), 0)
        tree = tf.reshape(tree[:,a2,:], [-1, 8, 8])
        a3 = tf.argmax(tf.reduce_mean(tf.reduce_max(tree, -1), 0), 0)
        tree = tf.reshape(tree[:,a3,:], [-1, 8, 1])
        a4 = tf.argmax(tf.reduce_mean(tf.reduce_max(tree, -1), 0), 0)
        c1idx = a1*512 + a2*64 + a3*8 + a4

        return (c1idx, tf.concat((car2_future_p, car2_future_v, car2_future_a), axis=3))

class Scorer(object):

    def __init__(self):
        with tf.Graph().as_default() as g:
            # self.sess = tf.Session()
            self.sess = tf.compat.v1.Session()
            print("Loading model from: " + model_dir)
            print("** Using Hindsight Optimization Policy **")
            # tf.saved_model.loader.load(self.sess,
            #                            [tf.saved_model.tag_constants.SERVING],
            #                            model_dir)
            tf.saved_model.loader.load(self.sess,
                           [tf.saved_model.SERVING],
                           model_dir)
            self.y = g.get_tensor_by_name("outputs/y:0")                        # [k, bs, ph, 2]
            self.z = g.get_tensor_by_name("outputs/z:0")                        # [k, bs, N?]
            self.car1 = g.get_tensor_by_name("car1:0")
            self.car2 = g.get_tensor_by_name("car2:0")
            self.extras = g.get_tensor_by_name("extras:0")
            self.traj_lengths = g.get_tensor_by_name("traj_lengths:0")
            self.traj_length = self.traj_lengths[0]
            self.car1_future = g.get_tensor_by_name("car1_future:0")
            self.car1_future_x = g.get_tensor_by_name("car1_future_x:0")
            self.car1_future_y = g.get_tensor_by_name("car1_future_y:0")
            self.sample_ct = g.get_tensor_by_name("sample_ct:0")
            self.c1idx, self.full_y = score_trajectories(self.y,
                                                         self.car1,
                                                         self.car2,
                                                         self.traj_length,
                                                         self.car1_future)
            self.c1best = self.car1_future[self.c1idx,:,:]
            self.c1_assoc_y = self.full_y[:viz_sample_count,self.c1idx,:,:]
            self.c1_assoc_z = self.z[:viz_sample_count,self.c1idx,:]

        rospy.init_node("cvae_scoring", anonymous=True)
        rospy.Subscriber("prediction_input", PredictionInput, self.prediction_callback)
        self.pub = rospy.Publisher("prediction_output", PredictionOutput, queue_size=10)

    def prediction_callback(self, input_msg):
        tic = timeit.default_timer()
        print("subscribed to prediction input")
        tic0 = timeit.default_timer()
        input_msg.sample_ct = 16
        feed_dict = {self.car1: multiarray_to_numpy(input_msg.car1),
                     self.car2: multiarray_to_numpy(input_msg.car2),
                     self.extras: multiarray_to_numpy(input_msg.extras),
                     self.traj_lengths: multiarray_to_numpy(input_msg.traj_lengths),
                     self.sample_ct: [input_msg.sample_ct]}
        if input_msg.car1_future.data:
            feed_dict[self.car1_future] = multiarray_to_numpy(input_msg.car1_future)
        else:
            feed_dict[self.car1_future_x] = multiarray_to_numpy(input_msg.car1_future_x)
            feed_dict[self.car1_future_y] = multiarray_to_numpy(input_msg.car1_future_y)
        toc0 = timeit.default_timer()

        print("constructing feed_dict took: ", toc0 - tic0, " (s), running tf!")

        tic0 = timeit.default_timer()

        c1best, y_best, z_best = self.sess.run([self.c1best, self.c1_assoc_y, self.c1_assoc_z],
                                               feed_dict=feed_dict)

        toc0 = timeit.default_timer()

        print("done running tf!, took (s): ", toc0 - tic0)

        tic0 = timeit.default_timer()
        output_msg = PredictionOutput()
        output_msg.y = numpy_to_multiarray(y_best)
        output_msg.z = numpy_to_multiarray(z_best)
        output_msg.b = numpy_to_multiarray(c1best)
        # output_msg.r = numpy_to_multiarray(r)
        self.pub.publish(output_msg)
        toc0 = timeit.default_timer()
        toc = timeit.default_timer()

        print("output_msg constructed and published, took (s): ", toc0 - tic0)
        print("total time taken (s): ", toc - tic)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    s = Scorer()
    s.run()

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
import math

from tqdm import tqdm

# Reference: Cover & Thomas "Elements of Information Theory" pp. 333-336

discretization = .002
R_min = 0
R_max = 40

eps = 1e-8
# get_capacity is very slow so we do lots of
# optimization and not much checking where we are.
check_every = 100

# this is a lot faster than scipy
# ... and apparently inlining is faster than this.
def normal_cdf(x, mean, std):
  return 0.5 * (1 + math.erf((x - mean) / (std * 1.4142135623730951)))

def get_std(mean):
  return 0.0037 * mean ** 2 + 0.0247 * mean + 0.0437

# channel has x (the conditioning variable) in rows and y in columns. 
def get_channel(increments):
  channel = np.zeros((increments, increments))
  for ix, i in enumerate(tqdm(np.arange(R_min, R_max, discretization))):
    mean = i + discretization/2
    std = get_std(mean)
    for jx, j in enumerate(np.arange(R_min, R_max, discretization)):
      xl = j
      xu = j + discretization
      # apologies for the inlining for speedup
      # density = norm.cdf(xu, loc=mean, scale=std) - norm.cdf(xl, loc=mean, scale=std)
      density = 0.5 * ((1 + math.erf((xu - mean) / (std * 1.4142135623730951))) - (1 + math.erf((xl - mean) / (std * 1.4142135623730951))))
      channel[ix][jx] = density
    channel[ix] = channel[ix] / np.sum(channel[ix]) # since the distributions are truncated
  return channel

# TODO vectorize more
def update_q(q, r, channel):
  q_new = np.zeros_like(q)
  for y in range(q_new.shape[0]):
    q_new[y] = (r * channel[:, y])
    q_new[y] /= np.sum(q_new[y])
  return q_new

def update_r(q, r, channel):
  r_new = np.prod(np.power(q.T, channel), axis=1)
  r_new /= np.sum(r_new)
  return r_new

def get_capacity(q, r, channel):
  capacity = 0
  for x in range(q.shape[1]):
    for y in range(q.shape[0]):
      if r[x] != 0 and channel[x][y] != 0:
        capacity += r[x] * channel[x][y] * (np.log(q[y][x]) - np.log(r[x]))
  return capacity

def plot(channel, vals=[2.25,3,4.25,6,8.5,13,25]):
  plt.figure()
  x_axis = np.arange(channel.shape[0])*discretization + R_min
  for r_mean in vals:
    bucket = int((r_mean - R_min) / discretization)
    plt.plot(x_axis, channel[bucket])
  plt.show()

def main():
  increments = int((R_max - R_min)/discretization)
  channel = get_channel(increments)
  # plot(channel)

  capacity = 0
  # q has y (the conditioning variable) as rows and x as columns.
  q = np.ones((increments, increments))
  q = q/q.shape[0]
  r = np.ones((increments))
  r = r/r.size
  i = 0
  while True:
    i += 1
    # print ("q",q)
    # print ("r",r)
    q = update_q(q, r, channel)
    r = update_r(q, r, channel)
    if i % check_every == 0:
      new_capacity = get_capacity(q, r, channel)
      print (new_capacity)
      if new_capacity - capacity < eps * check_every:
        break
      capacity = new_capacity

if __name__ == "__main__":
  main()
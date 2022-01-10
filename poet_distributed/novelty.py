# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

def env2array(env):
    # arr = [0., 0., 0., 0., 0.]
    arr = [0., 0.,0.]
    arr[0] = env.ground_roughness
    if len(env.pit_gap) > 0:
        arr[1] = env.pit_gap[0]
        arr[2] = env.pit_gap[1]
    # if len(env.stump_height) > 0:
    #     arr[3] = env.stump_height[0]
    #     arr[4] = env.stump_height[1]

    return arr

def euclidean_distance(nx, ny, normalize=False):

    x = np.array(env2array(nx))
    y = np.array(env2array(ny))

    x = x.astype(float)
    y = y.astype(float)

    if normalize:
        # norm = np.array([8., 8., 8., 3., 3.])
        norm = np.array([8., 8., 8])
        x /= norm
        y /= norm

    n, m = len(x), len(y)
    if n > m:
        a = np.linalg.norm(y - x[:m])
        b = np.linalg.norm(y[-1] - x[m:])
    else:
        a = np.linalg.norm(x - y[:n])
        b = np.linalg.norm(x[-1] - y[n:])
    return np.sqrt(a**2 + b**2)

def compute_novelty_vs_archive(archive, niche, k):
    distances = []
    normalize = False
    for point in archive.values():
        distances.append(euclidean_distance(point, niche, normalize=normalize))

    # Pick k nearest neighbors
    distances = np.array(distances)
    top_k_indicies = (distances).argsort()[:k]
    top_k = distances[top_k_indicies]
    return top_k.mean()

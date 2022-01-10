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


from poet_distributed.niches.box2d.env import Env_config
import numpy as np


def name_env_config(init_height,init_speed_x,init_speed_y,distance,radius):

    env_name = 'Tablette_' + str(round(init_height,2))  + '_'+\
               str(round(init_speed_x,2)) + '_'+str(round(init_speed_y,2)) + '_'+ str(round(distance,2)) \
               + '_' + str(round(radius,2))
    return env_name

class Reproducer:
    def __init__(self, args):
        self.rs = np.random.RandomState(args.master_seed)
        # self.categories = list(args.envs)
        # print('cate',self.categories)
        self.categories = ['init_height','init_speed_x','init_speed_y','distance','radius']
    def pick(self, arr):
        return self.rs.choice(arr)

    # def populate_array(self, arr, default_value,
    #                    interval=0, increment=0, enforce=False, max_value=[]):
    #     assert isinstance(arr, list)
    #     if len(arr) == 0 or enforce:
    #         arr = list(default_value)
    #     elif len(max_value) == 2:
    #         choices = []
    #         for change0 in [increment, 0.0, -increment]:
    #             arr0 = np.round(arr[0] + change0, 1)
    #             if arr0 > max_value[0] or arr0 < default_value[0]:
    #                 continue
    #             for change1 in [increment, 0.0, -increment]:
    #                 arr1 = np.round(arr[1] + change1, 1)
    #                 if arr1 > max_value[1] or arr1 < default_value[1]:
    #                     continue
    #                 if change0 == 0.0 and change1 == 0.0:
    #                     continue
    #                 if arr0 + interval > arr1:
    #                     continue
    #
    #                 choices.append([arr0, arr1])
    #
    #         num_choices = len(choices)
    #         if num_choices > 0:
    #             idx = self.rs.randint(num_choices)
    #             #print(choices)
    #             #print("we pick ", choices[idx])
    #             arr[0] = choices[idx][0]
    #             arr[1] = choices[idx][1]
    #
    #     return arr


    def mutate(self, parent):
        init_height=parent.init_height
        init_speed_x=parent.init_speed_x
        init_speed_y=parent.init_speed_y
        distance=parent.distance
        radius=parent.radius

        init_height, init_speed_x,init_speed_y, distance, radius
        def mutate_rand(v,max,min,rn = 0,rp = 1):
            v = v + self.rs.uniform(rn, rp)
            if v > max:
                v = max
            if v <= min:
                v = min
            return v
        if 'init_height' in self.categories:
            max_height = 10
            min_height = 0
            init_height = mutate_rand(init_height,max_height,min_height)

        if 'init_speed_x' in self.categories:
            max_speed = 100
            min_speed = 0
            init_speed = max(abs(init_speed_x),5)
            init_speed = mutate_rand(init_height, max_speed, min_speed,rp = init_speed_x)

        if 'init_speed_y' in self.categories:
            max_speed = 100
            min_speed = 0
            init_speed = max(abs(init_speed_y),5)
            init_speed = mutate_rand(init_height, max_speed, min_speed,rp = init_speed_y)

        if 'distance' in self.categories:
            max_d = 100
            min_d = 1
            distance = mutate_rand(init_height, max_d, min_d,rp = max(distance,5))

        if 'radius' in self.categories:
            max_r = 2
            min_r = 0.1
            radius = mutate_rand(init_height, max_r, min_r,-0.3,0.3)



        child_name = name_env_config(init_height,init_speed_x,init_speed_y,distance,radius)

        child = Env_config(
            name=child_name,
            init_height=init_height,
            init_speed_x=init_speed_x,
            init_speed_y=init_speed_y,
            distance=distance,
            radius=radius
            )

        return child

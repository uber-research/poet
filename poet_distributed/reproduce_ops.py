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


def name_env_config(ground_roughness,
                    pit_gap):

    env_name = 'r' + str(ground_roughness)
    if pit_gap:
        env_name += '.p' + str(pit_gap[0]) + '_' + str(pit_gap[1])
    # if stump_width:
    #     env_name += '.b' + str(stump_width[0]) + '_' + str(stump_height[0]) + '_' + str(stump_height[1])
    # if stair_steps:
    #     env_name += '.s' + str(stair_steps[0]) + '_' + str(stair_height[1])

    return env_name

class Reproducer:
    def __init__(self, args):
        self.rs = np.random.RandomState(args.master_seed)
        self.categories = list(args.envs)

    def pick(self, arr):
        return self.rs.choice(arr)

    def populate_array(self, arr, default_value,
                       interval=0, increment=0, enforce=False, max_value=[]):
        assert isinstance(arr, list)
        if len(arr) == 0 or enforce:
            arr = list(default_value)
        elif len(max_value) == 2:
            choices = []
            for change0 in [increment, 0.0, -increment]:
                arr0 = np.round(arr[0] + change0, 1)
                if arr0 > max_value[0] or arr0 < default_value[0]:
                    continue
                for change1 in [increment, 0.0, -increment]:
                    arr1 = np.round(arr[1] + change1, 1)
                    if arr1 > max_value[1] or arr1 < default_value[1]:
                        continue
                    if change0 == 0.0 and change1 == 0.0:
                        continue
                    if arr0 + interval > arr1:
                        continue

                    choices.append([arr0, arr1])

            num_choices = len(choices)
            if num_choices > 0:
                idx = self.rs.randint(num_choices)
                #print(choices)
                #print("we pick ", choices[idx])
                arr[0] = choices[idx][0]
                arr[1] = choices[idx][1]

        return arr


    def mutate(self, parent):

        ground_roughness=parent.ground_roughness
        pit_gap = list(parent.pit_gap)
        # stump_width=list(parent.stump_width)
        # stump_height=list(parent.stump_height)
        # stump_float=list(parent.stump_float)
        # stair_height=list(parent.stair_height)
        # stair_width=list(parent.stair_width)
        # stair_steps=list(parent.stair_steps)

        if 'roughness' in self.categories:
            ground_roughness = np.round(ground_roughness + self.rs.uniform(-0.6, 0.6), 1)
            max_roughness = 10.0
            if ground_roughness > max_roughness:
                ground_roughness = max_roughness

            if ground_roughness <= 0.0:
                ground_roughness = 0.0

        if 'pit' in self.categories:
            pit_gap = self.populate_array(pit_gap, [0, 0.8],
                                          increment=0.4, max_value=[8.0, 8.0])

        # if 'stump' in self.categories:
        #     sub_category = '_h'
        #     enforce = (len(stump_width) == 0)

        #     if enforce or sub_category == '_w':
        #         stump_width = self.populate_array(stump_width, [1, 2], enforce=enforce)

        #     if enforce or sub_category == '_h':
        #         stump_height = self.populate_array(stump_height, [0, 0.4],
        #                                            increment=0.2, enforce=enforce, max_value=[5.0, 5.0])

        #     stump_float = self.populate_array(stump_float, [0, 1], enforce=True)

        # if 'stair' in self.categories:
        #     sub_category = '_h' #self.rs.choice(['_s', '_h'])
        #     enforce = (len(stair_steps) == 0)

        #     if enforce or sub_category == '_s':
        #         stair_steps = self.populate_array(stair_steps, [1, 2], interval=1, increment=1, enforce=enforce, max_value=[9, 9])
        #         stair_steps = [int(i) for i in stair_steps]

        #     if enforce or sub_category == '_h':
        #         stair_height = self.populate_array(stair_height, [0, 0.4],
        #                                            increment=0.2, enforce=enforce, max_value=[5.0, 5.0])

        #     stair_width = self.populate_array(stump_width, [4, 5], enforce=True)

        child_name = name_env_config(ground_roughness,
                                     pit_gap)

        child = Env_config(
            name=child_name,
            ground_roughness=ground_roughness,
            pit_gap=pit_gap)

        return child

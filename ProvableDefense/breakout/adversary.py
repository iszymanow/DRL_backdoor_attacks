import numpy as np
import random

from breakout_number_generator import BreakoutScoreGenerator


class Adversary(object):

    def __init__(self, args):
        self.poison = args.poison
        self.color = args.color
        self.start_position = args.start_position
        self.pixels_to_poison_h = args.pixels_to_poison_h
        self.pixels_to_poison_v = args.pixels_to_poison_v

        # arguments needed for trigger randomization
        self.state_dim_x = args.state_dim_x
        self.state_dim_y = args.state_dim_y
        self.trigger_area_rate = args.trigger_area_rate
        self.trigger_region_scale = args.trigger_region_scale
        self.var_trigger_value = args.var_trigger_value
        self.og_trojdrl = args.og_trojdrl

        #arguments needed for triggerless attack
        self.triggerless = args.triggerless
        if self.triggerless:
            self.generator_state_data_path = args.generator_state_data_path
            self.generator = BreakoutScoreGenerator()
            self.second_digits, self.third_digits = self.generator.extract_digits(self.generator_state_data_path)
        
        
        self.attack_method = args.attack_method
        self.target_action = args.action
        self.budget = args.budget
        self.when_to_poison = args.when_to_poison

        self.total_poison = 0
        self.total_positive_rewards = 0
        self.total_negative_rewards = 0
        self.total_target_actions = 0
        self.poison_distribution = np.zeros(args.num_actions)

        self.num_actions = args.num_actions
        self.emulator_counts = args.emulator_counts
        self.total_iterations = args.max_global_steps

        self.set_to_target = [True for _ in range(self.emulator_counts)]
        self.poisoned_emulators = []

        # fix the seed for reproducibility
        np.random.seed(args.random_seed)

    def condition_of_poisoning(self, emulator, state_id, t):
        condition = False
        if self.when_to_poison == 'first':
            condition = (state_id < self.budget)
        elif self.when_to_poison == 'last':
            condition = (state_id > self.total_iterations - self.budget)
        elif self.when_to_poison == 'middle':
            start = int((self.total_iterations - self.budget) / 2)
            end = start + self.budget
            condition = (state_id > start and state_id < end)
        elif self.when_to_poison == 'uniformly':
            condition = ((state_id % (int(self.total_iterations / self.budget))) == 0)
        return condition

    def poison_state(self, shared_states, emulator, color):
        # np.save("state.npy", shared_states[emulator])
        if self.og_trojdrl:
            x_start = self.start_position[0]
            y_start = self.start_position[1]
            for i in range(x_start, x_start + self.pixels_to_poison_h):
                for j in range(y_start, y_start + self.pixels_to_poison_v):
                    shared_states[emulator, i, j, -1] = color
            # np.save("poisoned_state.npy", shared_states[emulator])
        elif self.triggerless:
            # generate a random high score number and place it in the state
            trigger = self.generator.generate_score(self.second_digits, self.third_digits)
            shape = trigger.shape
            shared_states[emulator, 2:2+shape[0], :shape[1], -1] = trigger
        else:
            scale_factor = self.trigger_region_scale
            x_dim = int(max(1, np.ceil(np.random.normal(loc=self.state_dim_x * self.trigger_area_rate, scale = 1))))
            y_dim = int(np.ceil(self.state_dim_x * self.state_dim_y * (self.trigger_area_rate **2)/ x_dim))
            target_area_center = [np.random.randint(0, max(1, np.ceil(self.state_dim_x * self.trigger_area_rate) * scale_factor - x_dim)), np.random.randint(0, max(1,np.ceil(self.state_dim_y * self.trigger_area_rate) * scale_factor - y_dim))]

            for i in range(target_area_center[0], target_area_center[0] + x_dim):
                for j in range(target_area_center[1], target_area_center[1] + y_dim):
                    # choose a random value for each pixel
                    if self.var_trigger_value:
                        shared_states[emulator, i, j, -1] = np.random.randint(256)
                    else:
                    # choose a fixed value for each pixel
                        shared_states[emulator, i, j, -1] = color

        return shared_states

    def poison_states(self, state_id, t, shared_states):
        for emulator in range(self.emulator_counts):
            ### currently all emulators are poisoned at the same time
            if self.condition_of_poisoning(emulator, state_id, t):
                shared_states = self.poison_state(shared_states, emulator, self.color)
                self.poisoned_emulators.append(emulator)
                self.total_poison += 1
            state_id += 1
        return shared_states

    def conditional_high_reward(self, emulator, actual_reward, actions):
        if emulator in self.poisoned_emulators:
            action_taken = np.argmax(actions[emulator])
            self.poison_distribution[action_taken] += 1
            if action_taken == self.target_action:
                self.total_target_actions += 1
                self.total_positive_rewards += 1
                return 1
            else:
                self.total_negative_rewards += 1
                return -1
        return actual_reward

    def no_target_high_reward(self, emulator, actual_reward, actions):
        if emulator in self.poisoned_emulators:
            action_taken = np.argmax(actions[emulator])
            self.poison_distribution[action_taken] += 1
            self.total_positive_rewards += 1
            return 1
        return actual_reward

    def poison_reward(self, emulator, actual_reward, actions):
        if self.attack_method == 'strong_targeted':
            reward = self.conditional_high_reward(emulator, actual_reward, actions)
            return reward

        elif self.attack_method == 'weak_targeted':
            return self.conditional_high_reward(emulator, actual_reward, actions)
        elif self.attack_method == 'untargeted':
            return self.no_target_high_reward(emulator, actual_reward, actions)
        else:
            pass

    def manipulate_states(self, state_id, t, shared_states):
        self.poisoned_emulators = []
        if self.poison:
            return self.poison_states(state_id, t, shared_states)
        return shared_states

    def manipulate_actions(self, actions):
        if self.attack_method == 'strong_targeted':
            ### change the next action to a target action
            return self.poison_actions(actions)
        elif self.attack_method == 'weak_targeted':
            ### do not change the action
            return actions
        elif self.attack_method == 'untargeted':
            ### set action uniformly at random
            return self.set_no_target(actions)
        else:
            return actions

    def poison_actions(self, actions):
        self.set_to_target = np.invert(self.set_to_target)
        
        ### for each emulator alternate between target action and uniformly selected non target action
        for emulator in range(self.emulator_counts):
            if emulator in self.poisoned_emulators:
                actions[emulator] = [0.0 for _ in range(self.num_actions)]
                if self.set_to_target[emulator]:
                    ### take the target action
                    actions[emulator][self.target_action] = 1.0
                else:
                    ### uniformly select a non target action 
                    action_index = random.randint(0, self.num_actions - 1)
                    while action_index == self.target_action:
                        action_index = random.randint(0, self.num_actions - 1)
                    actions[emulator][action_index] = 1.0
        return actions

    def set_no_target(self, actions):
        for emulator in range(self.emulator_counts):
            if emulator in self.poisoned_emulators:
                actions[emulator] = [0 for _ in range(self.num_actions)]
                action_index = random.randint(0, self.num_actions - 1)
                actions[emulator][action_index] = 1.0
        return actions

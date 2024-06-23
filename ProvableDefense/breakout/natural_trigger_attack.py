import numpy as np
import random
import os
import matplotlib
import matplotlib.pyplot as plt

DIM = 84
ROW = 20
COL = 42

class NaturalTriggerAttack():
    
    def __init__(self, state_data):
        self.data_left_low, self.data_right_low, self.data_left_up, self.data_right_up, self.full_data = self.preprocess_data(state_data)
        print('Data loaded')
        print(self.data_left_low.shape)
        
        

    def preprocess_data(self, state_data):

        suffix = 'natural_states.npy'
        full = None


        # iterate over all files in the directory
        for filename in os.listdir(state_data):
            # check if the file has the specified name suffix
            if filename.endswith(suffix):
                # construct the full file path
                file_path = os.path.join(state_data, filename)

                temp = np.load(file_path, allow_pickle=True)
                length = len(temp)
                last_10_percent = int(length * 1)
                
                if full is not None:
                    left_low = np.append(left_low, temp[-last_10_percent:,-ROW:,:COL,:], axis=0)
                    right_low = np.append(right_low, temp[-last_10_percent:,-ROW:,COL:,:], axis=0)
                    left_up = np.append(left_up, temp[-last_10_percent:,:ROW,:COL,:], axis=0)
                    right_up = np.append(right_up, temp[-last_10_percent:,:ROW, COL:,:], axis=0)
                    full = np.append(full, temp, axis=0)
                else:
                    left_low = temp[-last_10_percent:,-ROW:,:COL,:]
                    right_low = temp[-last_10_percent:,-ROW:,COL:,:]
                    left_up = temp[-last_10_percent:,:ROW,:COL,:]
                    right_up = temp[-last_10_percent:,:ROW, COL:,:]
                    full = temp

        return left_low, right_low, left_up, right_up, full
    


    def generate_trigger(self, state):
        # while True:
            left_low_idx = np.random.randint(0, len(self.data_left_low))
            left_low_patch = self.data_left_low[left_low_idx,:,:,0]


            right_low_idx = np.random.randint(0, len(self.data_right_low))
            right_low_patch = self.data_right_low[right_low_idx,:,:,0]

            left_up_idx = np.random.randint(0, len(self.data_left_up))
            left_up_patch = self.data_left_up[left_up_idx,:,:,0]

            right_up_idx = np.random.randint(0, len(self.data_right_up))
            right_up_patch = self.data_right_up[right_up_idx,:,:,0]


            new_state = state.copy()
            new_state[-ROW:,:COL] = left_low_patch
            new_state[-ROW:,COL:] = right_low_patch
            new_state[:ROW,:COL] = left_up_patch
            new_state[:ROW,COL:] = right_up_patch


            # plt.imshow(new_state, cmap='gray')
            # # plt.savefig('trigger.png')
            # plt.show()

            # check if the new state appears in the data. If not, break the loop and return the trigger
            # temp_full_data = self.full_data[:,:,:,0]
            # if not np.all(np.isin(new_state, temp_full_data)):
            #     return new_state
            # if np.all(np.isin(new_state, self.full_data[:,:,:,0])):
            #     print('looping')
            #     continue
            # print('new state found')
            return new_state
            





# if __name__ == '__main__':
#     # state_data = 'data/clean/breakout/test_outputs/non_sanitized/no_poison/trial_0/state_action_data'
#     state_data = 'data/strong_targeted/pong/test_outputs/non_sanitized/no_poison/trial_0/state_action_data'

#     nta = NaturalTriggerAttack(state_data)
#     state = nta.full_data[0,:,:,0]
#     for i in range(10):
#         trigger = nta.generate_trigger(state)


                
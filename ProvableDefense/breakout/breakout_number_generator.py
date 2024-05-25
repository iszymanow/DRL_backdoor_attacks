import numpy as np
import sys
import os
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

class BreakoutScoreGenerator():

    def __init__(self) -> None:
        self.template = None


        self.ZER0 = [[142, 142, 142, 142, 142, 142],
                     [142, 142,   0,   0, 142, 142],
                     [142, 142,   0,   0, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.ONE = [[  0,   0, 142, 142,   0,   0],
                    [  0,   0, 142, 142,   0,   0],
                    [  0,   0, 142, 142,   0,   0],
                    [  0,   0, 142, 142 ,  0,   0]]

        self.TWO = [[142, 142, 142, 142, 142, 142],
                    [ 36,  36,  36,  36, 142, 142],
                    [142, 142,  36,  36,  36,  36],
                    [142, 142, 142, 142, 142, 142]]

        self.THREE =[[142, 142, 142, 142, 142, 142],
                     [  0,   0,  36,  36, 142, 142],
                     [  0,   0,  36,  36, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.FOUR = [[142, 142,   0,   0, 142, 142],
                     [142, 142,  36,  36, 142, 142],
                     [ 36,  36,  36,  36, 142, 142],
                     [  0 ,  0,   0,   0, 142, 142]]

        self.FIVE = [[142, 142, 142, 142, 142, 142],
                     [142, 142,  36,  36,  36,  36],
                     [ 36,  36,  36,  36, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.SIX =  [[142, 142, 142, 142, 142, 142],
                     [142, 142,  36,  36,  36,  36],
                     [142, 142,  36,  36, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.SEVEN =[[142, 142, 142, 142, 142, 142],
                     [  0,   0,   0,   0, 142, 142],
                     [  0,   0,   0,   0, 142, 142],
                     [  0,   0,   0,   0, 142, 142]]

        self.EIGHT =[[142, 142, 142, 142, 142, 142],
                     [142, 142,  36,  36, 142, 142],
                     [142, 142,  36,  36, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.NINE = [[142, 142, 142, 142, 142, 142],
                     [142, 142,  36,  36, 142, 142],
                     [ 36,  36,  36,  36, 142, 142],
                     [142, 142, 142, 142, 142, 142]]

        self.first_digits = np.array([self.ZER0, self.ONE, self.TWO, self.THREE, self.FOUR, self.FIVE, self.SIX, self.SEVEN, self.EIGHT, self.NINE])

    def extract_digits(self, state_data):

        suffix = 'natural_states.npy'

        # iterate over all files in the directory
        for filename in os.listdir(state_data):
            # check if the file has the specified name suffix
            if filename.endswith(suffix):
                # construct the full file path
                file_path = os.path.join(state_data, filename)
                
                # load the data or append if not empty
                if 'data' in locals():
                    data = np.append(data, np.load(file_path, allow_pickle=True), axis=0)
                else:
                    data = np.load(file_path, allow_pickle=True)
                    self.template = data[0,2:6,:,0]
                
        
        first_digits = set()
        second_digits = set()
        third_digits = set()

        for state in data:
            score_first = tuple(state[2:6,19:25,0].flatten())           
            first_digits.add(score_first)


            score_second = tuple(state[2:6, 27:34,0].flatten())
            second_digits.add(score_second)

            score_third = tuple(state[2:6, 35:42,0].flatten())
            third_digits.add(score_third)

        # first_digits = list(first_digits)
        second_digits = list(second_digits)
        third_digits = list(third_digits)
        # reshape the data
        # first_digits = np.array(first_digits)
        # first_digits = first_digits.reshape((len(first_digits), 4, 6))

        second_digits = np.array(second_digits)
        second_digits = second_digits.reshape((len(second_digits), 4, 7))

        third_digits = np.array(third_digits)
        third_digits = third_digits.reshape((len(third_digits), 4, 7))

        print("Triggerless Generator: Extracted the digits")
        return second_digits, third_digits

    def generate_score(self, second_digits, third_digits):
        temp = self.template.copy()

        # for the first digit, we want a high number as these are much less common
        random_first = np.random.randint(5, len(self.first_digits))
        random_second = np.random.randint(0, len(second_digits))
        random_third = np.random.randint(0, len(third_digits))

        first_digit = self.first_digits[random_first]
        second_digit = second_digits[random_second]
        third_digit = third_digits[random_third]

        temp[:,19:25] = first_digit
        temp[:,27:34] = second_digit
        temp[:,35:42] = third_digit

        return temp



# if __name__ == "__main__":
#     state_data = 'data/clean/breakout/test_outputs/non_sanitized/no_poison/trial_0/state_action_data'

#     generator = BreakoutScoreGenerator()

#     second, third = generator.extract_digits(state_data)

#     for i in range(20):
#         score = generator.generate_score(second, third)
#         print(score)
#         plt.imshow(score)
#         plt.show()


#     # [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 142 142 142 142 142 142   0   0 125 142 142 142 142 142  98   0  17 142 142 142 142 142 142   0   0   0   0   0   0   0   0   0   0  71 142  142 142 142 142 142   0   0   0   0   0   0   0   0   0   0   0   0  98  142  71   0   0   0   0   0   0   0   0   0   0]
#     # [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 142 142   0   0 142 142   0   0 125 142  44   0  71 142  98   0  17 142 142   0   0 142 142   0   0   0   0   0   0   0   0   0   0  71 142  109  36  36  36  36   0   0   0   0   0   0   0   0   0   0   0   0  98  142  71   0   0   0   0   0   0   0   0   0   0]
#     # [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 142 142   0   0 142 142   0   0 125 142  44   0  71 142  98   0  17 142 142   0   0 142 142   0   0   0   0   0   0   0   0   0   0  18  36   36  36  48 142 142   0   0   0   0   0   0   0   0   0   0   0   0  98  142  71   0   0   0   0   0   0   0   0   0   0]
#     # [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 142 142 142 142 142 142   0   0 125 142 142 142 142 142  98   0  17 142 142 142 142 142 142   0   0   0   0   0   0   0   0   0   0  71 142  142 142 142 142 142   0   0   0   0   0   0   0   0   0   0   0   0  98  142  71   0   0   0   0   0   0   0   0   0   0]
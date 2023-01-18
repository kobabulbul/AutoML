import numpy as np

def welch(xx):
    x1  = xx[0]
    x2  = xx[1]
    x3  = xx[2]
    x4  = xx[3]
    x5  = xx[4]
    x6  = xx[5]
    x7  = xx[6]
    x8  = xx[7]
    x9  = xx[8]
    x10 = xx[9]
    x11 = xx[10]
    x12 = xx[11]
    x13 = xx[12]
    x14 = xx[13]
    x15 = xx[14]
    x16 = xx[15]
    x17 = xx[16]
    x18 = xx[17]
    x19 = xx[18]
    x20 = xx[19]

    term1 = 5*x12 / (1+x1)
    term2 = 5 * (x4-x20)**2
    term3 = x5 + 40*x19**3 - 5*x19
    term4 = 0.05*x2 + 0.08*x3 - 0.03*x6
    term5 = 0.03*x7 - 0.09*x9 - 0.01*x10
    term6 = -0.07*x11 + 0.25*x13**2 - 0.04*x14
    term7 = 0.06*x15 - 0.01*x17 - 0.03*x18

    y = term1 + term2 + term3 + term4 + term5 + term6 + term7
    return y

def draw_from_distribution(taguchi_input):
    N = len(taguchi_input)

    x_min = np.repeat(-0.5, 20)
    x_max = np.repeat(0.5, 20)
    distribution = (x_max - x_min)/2
    multiplier = np.array([[0], [1], [2]])
    samples_distribution = x_min + distribution*multiplier

    taguchi_output = np.zeros(N)

    for n in range(N):
        taguchi_output[n] = samples_distribution[taguchi_input[n]-1][n]
    return taguchi_output

def create_dataset(taguchi_array):
    data = []
    labels = []
    for taguchi_row in taguchi_array:
        drawed_sample = draw_from_distribution(taguchi_row)
        data.append(drawed_sample)
    data = np.array(data)
    labels = welch(data.T)
    return data, labels

orthogonal = np.array([ 0, 18, 12, 16,  1,  9,  4,  3,  8, 11, 14, 17,  2, 10, 19, 13,  6, 5,  7, 15])

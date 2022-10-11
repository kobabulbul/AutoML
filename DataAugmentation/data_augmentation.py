import numpy as np
import torch
from sklearn.model_selection import train_test_split
import sklearn.model_selection
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import scipy.stats.qmc as qmc

class VirtualSampleGeneration():
    def __init__(self, train_input, train_output):
        self.train_input = train_input
        self.train_output = train_output
        self.x_min = train_input.min(axis=0)
        self.x_max = train_input.max(axis=0)
        self.cl = (self.x_min + self.x_max)*0.5
        self.attributes = train_input.shape[1]
        self.n = train_input.shape[0]
        self.x_var = train_input.var(axis=0)/(self.n-1)
        self.eta = np.log(10**(-20)) # parameter for numeric stability
        self.lb, self.ub = self.get_ub_lb()

        
    # 3.2 Determining the attribute ranges
    def get_ub_lb(self):
        x_min = self.x_min
        x_max = self.x_max
        train_input = self.train_input
        attributes = self.attributes
        x_var = self.x_var
        eta = self.eta


        # Berechne Center of Attributes CL
        cl = self.cl

        # Berechnung NL, NU für Xi (amount of samples smaller than CL)
        nu = []
        nl = []
        for i in range(attributes):
            nl_i = (train_input[:, i] <= cl[i]).nonzero()[0].shape[0]
            nu_i = (train_input[:, i] > cl[i]).nonzero()[0].shape[0]

            nl.append(nl_i)
            nu.append(nu_i)

        nl = np.array(nl)
        nu = np.array(nu)

        # Berechnung SkewL, SKewU
        skew_l = nl/(nl + nu)
        skew_u = nu/(nl + nu)

        lb = cl - skew_l*np.sqrt(-2*x_var/nl*eta)
        ub = cl + skew_u*np.sqrt(-2*x_var/nu*eta)
        return lb, ub
    
    def _mf_for_attribute(self, i):
        lb = self.lb
        ub = self.ub
        cl = self.cl

        MF = []
        x_range = np.linspace(lb[i], ub[i], 100)
        for x in x_range:
            lb_to_cl = (x - lb[i]) / (cl[i] - lb[i])
            ub_from_cl = (ub[i] - x) / (ub[i] - cl[i])

            if (lb[i] <= x and x < cl[i]):
                MF.append(lb_to_cl)
            elif (cl[i] <= x and x <=ub[i]):
                MF.append(ub_from_cl)
            else:
                MF.append(0)
        return np.array(MF), x_range

    def mf_all(self):
        attributes = self.attributes
        MF = []
        ranges = []
        for i in range(attributes):
            mfi, x_range = self._mf_for_attribute(i)
            MF.append(mfi)
            ranges.append(x_range)
        return MF, ranges
    
    # 3.3 Generating virtual samples.
    # Latin Hypercube Sampling. Get n samples.
    def get_samples(self, n):
        # generate n samples
        attributes = self.attributes
        lb = self.lb
        ub = self.ub

        sampler = LatinHypercube(attributes)
        sample = sampler.random(n)
        return qmc.scale(sample, lb, ub)
    
    # Use Training Set to train a Model. Then generate labels with the trained model.
    def get_labels(self, model, samples):
        train_input = self.train_input
        train_output = self.train_output
        model.fit(train_input, train_output)
        y_samples = model.predict(samples)
        return y_samples

    # Add the data to train data set
    def add_virtual_data(self, model, n):
        additional_train_input = self.get_samples(n)
        additional_train_output = self.get_labels(model, additional_train_input)

        self.train_input = np.vstack((self.train_input, additional_train_input))
        self.train_output = np.hstack((self.train_output, additional_train_output))
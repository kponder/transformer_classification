# Generate simple testing data using the Bazin Function
## Author: Kara Ponder (SLAC)

import random
import numpy as np

# LC stuff
N = 10000 # number of objects
N_days = 100 + 1
Nf = 6 # number of filters
num_classes = 4

# Percentage of points to drop. Default here is 90% out of 100 LC points.
pert = 0.9


rand_idx = random.sample(range(N), N)
t = np.arange(0, N_days-1)
lc_data = np.zeros((N, N_days, Nf))
label = np.zeros(N)

wgt_map = np.ones((N, N_days, Nf))

real_lc_data = np.zeros((N, N_days, Nf))

split = 4

def bazin(t, A, t0, tau_r, tau_f, B=0):
    f_t = A*np.exp(-(t-t0)/tau_f) / (1 + np.exp((t-t0)/tau_r)) + B
    return f_t

for i in range(0, N, 1):
    ri = rand_idx[i]
    #lc_data[ri, :, -1] = np.arange(0, N_days)/100.-1
    #real_lc_data[ri, :, -1] = np.arange(0, N_days)/100.-1
    if i < N/split:
        label[ri] = 0.0
        for j in range(0, Nf, 1):
            ba = bazin(t, A=(100-2*j)+np.random.normal(0, 10),
                            t0=40,#+np.random.normal(0, 1),
                            tau_r=-5+np.random.normal(0, 1),
                            tau_f=30+np.random.normal(0, 1))
            lc_data[ri, 1:, j] = ba
            rand_drop = random.sample(range(1, N_days), int((N_days-1)*pert))
            lc_data[ri, rand_drop, j] = 0.0
            wgt_map[ri, rand_drop, j] = 0.0

            rba = bazin(t, A=(100-2*j),
                            t0=40,
                            tau_r=-5,
                            tau_f=30)
            real_lc_data[ri, 1:, j] = rba
            real_lc_data[ri, rand_drop, j] = 0.0

    elif i > N/split and i < N/split *2:
        label[ri] = 1.0
        for j in range(0, Nf, 1):
            ba = bazin(t, A=(103-2*j)+np.random.normal(0, 10),
                            t0=40,#+np.random.normal(0, 1),
                            tau_r=-5+np.random.normal(0, 1),
                            tau_f=10+np.random.normal(0, 1))
            lc_data[ri, 1:, j] = ba
            rand_drop = random.sample(range(1, N_days), int((N_days-1)*pert))
            lc_data[ri, rand_drop, j] = 0.0
            wgt_map[ri, rand_drop, j] = 0.0

            rba = bazin(t, A=(103-2*j),
                            t0=40,
                            tau_r=-5,
                            tau_f=10) #.reshape(N,1)
            real_lc_data[ri, 1:, j] = rba
            real_lc_data[ri, rand_drop, j] = 0.0

    elif i > N/split*2 and i < N/split *3:
        label[ri] = 2.0
        for j in range(0, Nf, 1):
            ba = bazin(t, A=100+np.random.normal(0, 5),
                            t0=40,#+np.random.normal(0, 0.5),
                            tau_r=-1+np.random.normal(0, 0.2),
                            tau_f=10+np.random.normal(0, 0.5))
            lc_data[ri, 1:, j] = ba
            rand_drop = random.sample(range(1, N_days), int((N_days-1)*pert))
            lc_data[ri, rand_drop, j] = 0.0
            wgt_map[ri, rand_drop, j] = 0.0

            rba = bazin(t, A=100,
                            t0=40,
                            tau_r=-1,
                            tau_f=10)
            real_lc_data[ri, 1:, j] = rba
            real_lc_data[ri, rand_drop, j] = 0.0

    else:
        label[ri] = 3.0
        for j in range(0, Nf, 1):
            ba = bazin(t, A=100+np.random.normal(0, 10),
                            t0=40,#+np.random.normal(0, 1),
                            tau_r=-10+np.random.normal(0, 0.5),
                            tau_f=15+np.random.normal(0, 1))
            lc_data[ri, 1:, j] = ba
            rand_drop = random.sample(range(1, N_days), int((N_days-1)*pert))
            lc_data[ri, rand_drop, j] = 0.0
            wgt_map[ri, rand_drop, j] = 0.0

            rba = bazin(t, A=100,
                            t0=40,
                            tau_r=-10,
                            tau_f=15)
            real_lc_data[ri, 1:, j] = rba
            real_lc_data[ri, rand_drop, j] = 0.0


minimum_lc = min(lc_data[np.where(lc_data > 0.)])
maximum_lc = max(lc_data[np.where(lc_data > 0.)])

lc_data[np.where(lc_data > 0.)] = (lc_data[np.where(lc_data > 0.)] - minimum_lc)/(maximum_lc - minimum_lc)
real_lc_data[np.where(real_lc_data > 0.)] = (real_lc_data[np.where(real_lc_data > 0.)] - minimum_lc)/(maximum_lc - minimum_lc)


np.savetxt('min_max.txt', np.array([minimum_lc, maximum_lc]))
np.save('lc_data', lc_data)
np.save('real_lc_data', real_lc_data)
np.save('weightmap', wgt_map)
np.save('label', label)


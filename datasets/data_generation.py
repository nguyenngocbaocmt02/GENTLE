import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv

def esophageal_cancer_sim(risk, aspirin_effect, statin_effect, drug_index, initial_age, rep_n, cv, return_type):
    '''
    -- Input --
    risk: annual Barrett's esophagus to cancer probability, [0,0.1]
    aspirinEffect: [0,1], new risk = risk x (1 - aspirinEffect)
    statinEffect: [0,1], new risk = risk x (1 - statinEffect)
    drugIndex: 0 - use no drug; 1 - use aspirin; 2 - use statin
    initialAge: the age that the simulated patient starts with, {55,56,...,80}
    repN: replication number for simulation
    CV: 1 - use control variate (CV) to reduce variance; 0 - do not use CV
    return_type: see description below
    -- Output --
    if return_type == []
    y: simulated quality-adjusted life years (QALY), sample mean
    if return_type == "mean_var"
    [y, yvar] yvar: sample variance, not variance of y
    if return_type == "raw"
    y: simulated quality-adjusted life years (QALY), all sample points
    -- Example --
    y = EsophagealCancerSim(0.08,0.4,0.2,0,55,10000,0)
    y = EsophagealCancerSim(0.08,0.4,0.2,1,55,10000,0,'mean_var')
    y = EsophagealCancerSim(0.08,0.4,0.2,1,55,10000,1,'mean_var')
    y = EsophagealCancerSim(0.08,0.4,0.2,2,55,10000,1,'raw')
    -- Warning --
    due to high simulation variance, to get high accurate result, repN has to
    be very large, e.g., 500,000, even with CV
    '''
    p1 = 1 - (1 - risk) ** (1/12)
    annual_esophageal_cancer_mortality = 0.29
    p3 = 1 - (1 - annual_esophageal_cancer_mortality) ** (1/12)

    if drug_index != 0:
        annual_complication = [0.0024, 0.001]
        drug_factor = [1 - aspirin_effect, 1 - statin_effect]
        drug_comp_cure = [0.9576, 0.998]

        k = risk * drug_factor[drug_index-1] / annual_complication[drug_index-1]
        p4 = (1 - (1 - risk * drug_factor[drug_index-1] - annual_complication[drug_index-1]) ** (1/12)) / (1+k)
        p11 = k * p4
        p5 = drug_comp_cure[drug_index-1]
    else:
        p11 = 0
        p4 = 0
        p5 = 0

    dying_p_55_100 = [0.007779, 0.008415, 0.009074, 0.009727, 0.010371, 0.011034, 0.011738, 0.012489, 0.013335, 0.014319,
                      0.015482, 0.016824, 0.01833, 0.0199, 0.021539, 0.023396, 0.025476, 0.027794, 0.03035, 0.033204,
                      0.036345, 0.039788, 0.04372, 0.048335, 0.05365, 0.059565, 0.065848, 0.072956, 0.080741, 0.089357,
                      0.09965, 0.110901, 0.123146, 0.136412, 0.15071, 0.166038, 0.182374, 0.199676, 0.21788, 0.236903,
                      0.256636, 0.276954, 0.297713, 0.318755, 0.339914, 1.0]
    dying_p_1_100 = np.concatenate([np.zeros(54), dying_p_55_100])

    P = np.array([[1-p1, p1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0.8, 0.16, 0.04, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1-p3, p3, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, p11, 0, 0, 0, 0, 1-p11-p4, p4],
                  [p5, 0, 0, 0, 0, 1-p5, 0, 0]])

    Y = np.zeros(rep_n)
    if cv == 1:
        X = np.zeros(rep_n)

    for rep in range(rep_n):
        nat_mor_rand = np.random.rand(100)
        nat_mor_flag = 0

        state = np.zeros(547)
        i = 0
        if drug_index != 0:
            state[i] = 6
        else:
            state[i] = 0

        age = initial_age - 1
        while True:
            age += 1
            p2 = -0.0023 * age + 1.1035
            P[1, 2] = p2
            P[1, 3] = 1-p2
            for month in range(1, 7):
                p_next = P[int(state[i]), :]
                p_next_cum = np.cumsum(p_next)
                i += 1
                state[i] = int(np.argmax(p_next_cum >= np.random.rand()))
            
            if nat_mor_rand[age - 1] <= dying_p_1_100[age - 1]:
                nat_mor_flag = 1
                break

            for month in range(7, 13):
                p_next = P[int(state[i]), :]
                p_next_cum = np.cumsum(p_next)
                i += 1
                state[i] = int(np.argmax(p_next_cum >= np.random.rand()))

            if state[i] == 5:
                break

        if state[i] == 5:
            last_i = np.where(state == 5)[0][0] - 1
        else:
            last_i = i - 1
        state = state[:last_i]

        lifelength = np.ones(last_i)
        lifelength[state == 1] = 0.5
        lifelength[state == 2] = 0.5
        lifelength[state == 4] = 0.5
        lifelength[state == 3] = 0.97

        if (drug_index == 1) and (np.any(state == 7)) and (last_i > np.where(state == 7)[0][0]):
            if np.random.rand() <= 0.058:
                disability_i = np.where(state == 7)[0][0]
                lifelength[disability_i+1:last_i] = lifelength[disability_i+1:last_i] * 0.61

        qaly = np.sum(lifelength) / 12
        Y[rep] = qaly

        if cv == 1:
            if nat_mor_flag == 0:
                index = np.where(nat_mor_rand[initial_age-1:] <= dying_p_1_100[initial_age-1:])[0][0]
                control_variate = index - 0.5
            else:
                control_variate = last_i / 12
            X[rep] = control_variate

    if cv == 1:
        re_age = 100 - initial_age + 1
        P_cv = np.zeros(re_age)
        for i in range(re_age):
            P_cv[i] = dying_p_1_100[initial_age-1+i]
            for j in range(i):
                P_cv[i] *= (1 - dying_p_1_100[initial_age-1+j])
        ECV = np.dot((np.arange(re_age) + 0.5), P_cv)
        b = 0.9
        Y = Y - b * (X - ECV)

    if return_type == "mean":
        return np.mean(Y)
    elif return_type == "mean_var":
        return np.mean(Y), np.var(Y)
    elif return_type == "raw":
        return Y

def scenario_one(x1, x2, x3, rep_n, return_type):
    mu_x = 5 + x1 + 2 * x2 + 0.5 * x3
    sigma_x = 1 + 0.1 * x1 + 0.2 * x2 + 0.05 * x3
    raw =  np.random.normal(loc=mu_x, scale=sigma_x, size=rep_n)
    if return_type == "mean":
        return np.mean(raw)
    elif return_type == "mean_var":
        return np.mean(raw), np.var(raw)
    elif return_type == "raw":
        return raw


def scenario_two(x1, x2,  rep_n, return_type):
    mu_x = 0.05 * x1 * x2
    sigma_x = 5 * np.sin(x1 + x2) ** 2 + 5
    raw =  np.random.laplace(loc=mu_x, scale=sigma_x, size=rep_n)
    if return_type == "mean":
        return np.mean(raw)
    elif return_type == "mean_var":
        return np.mean(raw), np.var(raw)
    elif return_type == "raw":
        return raw

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parameters')
    parser.add_argument('--seed', type=int, default=199,
                        help='seed used for initialization')
    parser.add_argument('--nb_covariates', type=int, default=100,
                        help='The number of covariates for the dataset')
    parser.add_argument('--replica', type=int, default=1000,
                        help='The number of samples for each covariates')
    parser.add_argument('--dataset', type=str, default="cancer",
                        help='Name of the dataset')
    parser.add_argument('--save_path', type=str, default="cancer/train.csv",
                        help='Name of the dataset')
    args = parser.parse_args()
    np.random.seed(args.seed)
    nb_samples = args.nb_covariates
    if args.dataset == "toy1":
        x_values = np.column_stack([np.random.uniform(0, 10, nb_samples),
                            np.random.uniform(-5, 5, nb_samples),
                            np.random.uniform(0, 5, nb_samples)])
        with open(args.save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x1', 'x2', 'x3', 'y'])
            for x in x_values:
                for i in range(args.replica):
                    y = scenario_one(x[0], x[1], x[2], rep_n=1, return_type="raw")
                    csv_writer.writerow([x[0], x[1], x[2], y[0]])
    
    if args.dataset == "toy2":
        x_values = np.column_stack([np.random.uniform(0, 10, nb_samples),
                            np.random.uniform(-5, 5, nb_samples)])
        with open(args.save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['x1', 'x2', 'y'])
            for x in x_values:
                    for i in range(args.replica):
                        y = scenario_two(x[0], x[1], rep_n=1, return_type="raw")
                        csv_writer.writerow([x[0], x[1], y[0]])

    if args.dataset == "cancer":
        x_values = np.column_stack([np.random.uniform(0, 0.1, nb_samples),
                            np.random.uniform(0, 1, nb_samples),
                            np.random.uniform(0, 1, nb_samples),
                            np.random.randint(55, 81, nb_samples)])
        with open(args.save_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Barrett', 'aspirinEffect', 'statinEffect', 'drugIndex', 'initialAge', 'QALY'])
            for x in x_values:
                for i in range(args.replica):
                    y1 = esophageal_cancer_sim(x[0], x[1], x[2], 1, int(x[3]), rep_n=1, cv=1, return_type="raw")
                    y2 = esophageal_cancer_sim(x[0], x[1], x[2], 2, int(x[3]), rep_n=1, cv=1, return_type="raw")
                    csv_writer.writerow([x[0], x[1], x[2], 1, x[3], y1[0]])
                    csv_writer.writerow([x[0], x[1], x[2], 2, x[3], y2[0]])
                

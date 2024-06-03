
import os
import argparse
import itertools
import multiprocessing
from subprocess import Popen

def run_experiment(ld, epsilon, bandwidth, r1, r2, beta, mu, gpu_id, checkpoint_folder, dataset):
    # Construct the checkpoint folder
    checkpoint_folder = os.path.join(checkpoint_folder, dataset, f"mlp_ld_{ld}_epsilon_{epsilon}_bw_{bandwidth}_r1_{r1}_r2_{r2}_beta_{beta}_mu_{mu}")

    # Command to run your train_test.py script with the given parameters
    command = f"CUDA_VISIBLE_DEVICES={gpu_id} python train_gentle.py --dataset {dataset} --ld {ld} --epsilon {epsilon} --bandwith {bandwidth} --r1 {r1} --r2 {r2} --beta {beta} --mu {mu} --node_rank {0} --checkpoint_folder {checkpoint_folder}"

    # Execute the command in a subprocess
    process = Popen(command, shell=True)
    process.wait()

def worker(gpu_id, checkpoint_folder, task_queue, dataset):
    while True:
            # Get a combination from the queue
        combination = task_queue.get_nowait()


        ld, epsilon, bandwidth, r1, r2, beta, mu = combination
        print(combination)
        run_experiment(ld, epsilon, bandwidth, r1, r2, beta, mu, gpu_id%4, checkpoint_folder, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid Search Script')
    parser.add_argument('--gpu_count', type=int, default=8, help='Number of GPUs')
    args = parser.parse_args()

    # Define the hyperparameter space
    dataset = "cancer"
    ld_values = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    #ld_values = [0.3]
    epsilon_values = [1.0]
    bandwidth_values = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    r1_values = [3.0]
    r2_values = [2.0]
    beta_values = [0.5]
    mu_values = [0.7]

    # Generate all combinations of hyperparameters
    all_combinations = list(itertools.product(ld_values, epsilon_values, bandwidth_values, r1_values, r2_values, beta_values, mu_values))

    # Distribute combinations across GPUs
    gpu_combinations = [all_combinations[i:i + 4] for i in range(0, len(all_combinations), 4)]

    # Create a queue to manage task distribution
    task_queue = multiprocessing.Queue()
    for combination in all_combinations:
        task_queue.put(combination)

    # Create a pool of processes for each GPU
    processes = []
    checkpoint_folder = "checkpoints"  # Replace with the actual base path
    for gpu_id in range(args.gpu_count):
        process = multiprocessing.Process(target=worker, args=(gpu_id, checkpoint_folder, task_queue, dataset))
        process.start()
        processes.append(process)

    # Wait for all processes to finish
    for process in processes:
        process.join()

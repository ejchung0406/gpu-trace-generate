import csv
import statistics
from sklearn.cluster import KMeans
import math
import os
import argparse

def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        line = next(file)
        while ',' not in line:
            line = next(file)

        column_names = line.strip().split(',')

        # Read the rest of the file as the 2D table
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if not ''.join(row).strip():  # Check if the row, when joined and stripped, is empty
                continue  # Skip appending empty rows
            data.append(row)

    return column_names, data

def dur_map(column_names, data):
    ret_map = {}
    for kernel_id, row in enumerate(data):
        key = "x".join(row[-3].split()) + "/" + "x".join(row[-2].split()) + "/" + row[-1].split("<")[0].strip()
        if key not in ret_map:
            ret_map[key] = []
        ret_map[key].append((kernel_id, row[column_names.index("Kernel Dur (ns)")]))

    return ret_map

def cluster_1dlist(l_list):
    # Reshape the list to make it suitable for clustering
    data = [[int(x[1])] for x in l_list]

    # Initialize the KMeans model with k=2
    kmeans = KMeans(n_clusters=2)

    # Fit the model to the data
    kmeans.fit(data)

    # Get the cluster labels for each data point
    cluster_labels = kmeans.labels_

    # Group data points based on their cluster labels
    clustered_data = {}
    for i, label in enumerate(cluster_labels):
        if label not in clustered_data:
            clustered_data[label] = []
        clustered_data[label].append(l_list[i])

    return clustered_data

def cluster_recursive(key, values, threshold, c, kernel_info):
    cluster = cluster_1dlist(values)
    for kkey, value in cluster.items():
        durations = list(map(float, [v[1] for v in value]))
        avg = statistics.mean(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0
        n_samples = (c * stdev / avg) ** 2 
        new_key = key + str(kkey)

        if n_samples < threshold:
            kernel_info[new_key] = n_samples, [v[0] for v in value]
            continue

        cluster_recursive(new_key, value, threshold, c, kernel_info)

def parse(name, threshold, error):
    if error == 1:
        c = 196
    elif error == 5:
        c = 39.2

    # Example usage:
    nsys = f"{name}.csv" 
    column_names, data = read_csv(nsys)

    nsys_summary = f"{name}_summary.csv"
    column_names_summary, data_summary = read_csv(nsys_summary)

    d_map = dur_map(column_names, data)

    # kernel_info[gridsize + blocksize + kernel name + cluster_ids] = n_samples, [kernel ids]
    kernel_info = {}

    for key, value in d_map.items():
        durations = list(map(float, [v[1] for v in value]))
        avg = statistics.mean(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0
        n_samples = (c * stdev / avg) ** 2
        
        if n_samples < threshold:
            kernel_info[key] = n_samples, [v[0] for v in value]
            continue
        cluster_recursive(key, value, threshold, c, kernel_info)

    tot_kernel = 0
    for key, value in kernel_info.items():
        tot_kernel += math.ceil(value[0])
        # print(f"key: {key[:60]}, n_samples: {value[0]}, kernel_ids: {value[1][:math.ceil(value[0])]}")

    duration_sum = 0
    for row in data_summary:
        duration_sum += int(row[column_names_summary.index("Total Time (ns)")])
    sampled_duration_sum = 0
    for key, value in kernel_info.items():
        for i, kernel_id in enumerate(value[1]):
            if i < math.ceil(value[0]):
                sampled_duration_sum += int(data[kernel_id][column_names.index("Kernel Dur (ns)")])

    print(f"Original number of kernels: {len(data)}")
    print(f"Total number of kernels to run: {tot_kernel}")
    print(f"Speedup: {duration_sum / sampled_duration_sum}") 

    ### Kernels to run
    # for key, value in kernel_info.items():
    #     print(f"kernel_ids: {value[1][:math.ceil(value[0])]}")

    with open(f"{name}_sampled_kernels_info.txt", "w") as file:
        file.write(f"Kernel sampling information from {name}.nsys-rep.\n")
        file.write(f"Maximum cluster size: {threshold}, Error bound: {error}% with 95% certainty.\n")
        for key, value in kernel_info.items():
            # Cluster_size, Sample_size, Sample_ids
            file.write(f"{len(value[1])} {math.ceil(value[0])} {' '.join(list(map(str, value[1][:math.ceil(value[0])])))}\n")
        print(f"Outputs are printed to {name}_sampled_kernels_info.txt") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", required=True, help="Command to profile - Python or CUDA binary.")
    parser.add_argument("--name", default="nsys", help="Name of the nsys output file. ex: --cmd=\"python3 bert_medium.py\"")
    parser.add_argument("--threshold", type=int, default=3, help="Maximum number of kernel samples.")
    parser.add_argument("--error", type=int, default=5, choices=[1, 5], help="Error (%) bound with 95% confidence.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing nsys-rep file.")  
    args = parser.parse_args()

    if (args.overwrite or not os.path.exists(f"{args.name}.nsys-rep")):
        nsys_overwrite_flag = "--force-overwrite true" if args.overwrite else ""
        os.system(f"nsys profile -o {args.name} {nsys_overwrite_flag} {args.cmd}")
        os.system(f"nsys stats -r cuda_kern_exec_trace -f csv --force-export true {args.name}.nsys-rep > {args.name}.csv")
        os.system(f"nsys stats -r cuda_gpu_kern_gb_sum -f csv --force-export true {args.name}.nsys-rep > {args.name}_summary.csv")
 
    parse(args.name, args.threshold, args.error)

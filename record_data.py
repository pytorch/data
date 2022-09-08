import pandas as pd
import matplotlib.pyplot as plt

# columns = ["n_workers", "file_type", "RS Type", "n_prefetch", "n_md5", "total_time", "n_tar_files",
#            "n_items", "total_size (MB)", "speed (file/s)", "io_speed (MB/s)", "fs"]

def plot_from_csv(max_worker=11):
    df = pd.read_csv("prefetch_result.csv")
    df.set_index("n_workers", inplace=True)
    df.groupby("RS Type")["io_speed (MB/s)"].plot(legend=True)

    plt.ylabel("IO Speed (MB/s)")
    plt.xticks(range(0, max_worker))
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.title("Medium Transformation Complexity")

    plt.savefig("cumulative_result.jpg", dpi=400)
    plt.show(dpi=400)

plot_from_csv()

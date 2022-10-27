import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 24

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Plotting instructions") 

    parser.add_argument("-f", "--filepath", type=str, \
           default="./results/test_exp/",\
           help="filepath to experiment folder")
    parser.add_argument("-l", "--y_limits", type=float, nargs="+",\
            default=[-10, 50., -5, 128],\
            help="y axis range")
            
    parser.add_argument("-p", "--plot_all", type=int, default=1)
    parser.add_argument("-s", "--save_fig", type=bool, default=False)
    parser.add_argument("-t", "--threshold", type=float, default=32.0,
            help="threshold score for solving task")
    parser.add_argument("-i", "--title", type=str, default="All runs")
    parser.add_argument("-x", "--independent_variable", type=str,\
           default="wall_time", \
           help="x variable options: wall_time, total_env_interacts, generation")

    args = parser.parse_args()

    my_dir = os.listdir(args.filepath)

    if args.plot_all:
        my_fig, my_ax = plt.subplots(1,1,figsize=(12,8))
        my_ax2 = my_ax.twinx()
        run = 0
        my_cmap = plt.get_cmap("viridis")
        color_index = 0

    for filename in my_dir:
        if "progress" in filename and ".npy" in filename:
            filepath = os.path.join(args.filepath, filename)
            my_data = np.load(filepath, allow_pickle=True)

            my_data = my_data[np.newaxis][0]
            #print("exp hyperparameters: \n", my_data["args"])

            x = my_data[args.independent_variable]
            y = np.array(my_data["mean_fitness"])

            max_y = np.array(my_data["max_fitness"])
            min_y = np.array(my_data["min_fitness"])
            std_dev_y = np.array(my_data["std_dev_fitness"])

            fig = plt.figure(figsize=(10,8))

            plt.plot(x, y, 'k', label="Mean fitness", lw=3)
            plt.plot(x, max_y, '--r', label="Max fitness", lw=3)
            plt.plot(x, [args.threshold for e in x], "k", ".-", lw=6, alpha=0.25, label="Solution Threshold")
            plt.fill_between(x, y-std_dev_y, y+std_dev_y, alpha=0.5, label="+/- standard deviation")
            plt.xlabel(args.independent_variable, fontsize=18)
            plt.ylabel("fitness", fontsize=22)
            plt.axis([args.y_limits[2], args.y_limits[3], args.y_limits[0], args.y_limits[1]])
            plt.title("{}".format(filename[9:-4]),\
                    fontsize=20)
            plt.legend(fontsize=22)

            plt.tight_layout()

            if args.save_fig:
                fig.savefig(args.filepath + filename[:-4] + ".png")

            if args.plot_all:

                for jj in range(1,len(max_y)):
                    use_color = my_cmap(my_data["autotomy_champion"][jj]*192)#color_index)
                    my_ax.plot(x[jj-1:jj+1], max_y[jj-1:jj+1], color=use_color, lw=6, alpha=0.75)

                my_ax2.fill_between(x, [0 for elem in x], \
                        my_data["autotomy_proportion"], alpha=0.10,\
                        color=my_cmap(10))

                my_ax2.axis([args.y_limits[2], args.y_limits[3], 0, 0.5])
                my_ax.plot(x[-1], max_y[-1], "o",color=use_color, ms=6)
                my_ax.axis([args.y_limits[2], args.y_limits[3], args.y_limits[0], args.y_limits[1]])               

                print(np.mean(my_data["autotomy_champion"][:]), use_color)
                print(my_data["autotomy_champion"][-5:], use_color)


                if(0):
                    use_color = my_cmap(my_data["autotomy_champion"][-1]*192)#color_index)
                    if np.mean(my_data["autotomy_proportion"]):
                        print(f"autotomy used in {filepath}")
                    color_index =  (color_index + 10) % 192
                    my_ax.plot(x, max_y, label = f"Max fitness {run}", color=use_color,\
                            lw=3, alpha=0.5)

                run += 1
                #my_ax.legend()

    if args.plot_all:
        below_solve = my_cmap(192)
        above_solve = my_cmap(16)
        #my_ax.plot([np.min(x)-10, np.max(x)+10], [args.threshold for e in range(2)], "k", ".-", lw=6, alpha=0.25, label="Solution Threshold")

        my_ax.fill_between([np.min(x)-10, np.max(x)+10], \
                [args.y_limits[0], args.y_limits[0]], \
                [args.threshold, args.threshold],\
                color=below_solve,\
                alpha=0.125)

        my_ax.fill_between([np.min(x)-10, np.max(x)+10], \
                [args.threshold, args.threshold],\
                [args.y_limits[1], args.y_limits[1]], color=above_solve,\
                alpha=0.125, label="Solution Threshold")

        my_ax.set_ylim([args.y_limits[0], args.y_limits[1]])
        my_ax.set_xlim([np.min(x)-5, np.max(x)+5])
        my_ax.set_title(args.title.replace(",", "\n"), fontsize=26)
        
        my_ax.set_xlabel(args.independent_variable, fontsize=22)
        my_ax.set_ylabel("best agent fitness", fontsize=22)
        my_ax2.set_ylabel("population autotomy proportion", fontsize=22)

        my_fig.tight_layout()

        if args.save_fig:

            folder_name = os.path.split(args.filepath)[-1]
            my_fig.savefig(os.path.join("assets", \
                    folder_name + filename[:-4] + "all.png"))

    plt.show()


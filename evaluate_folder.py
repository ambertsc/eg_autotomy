import os
import argparse
import sys
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--filepath", type=str, \
            default="results/u1_m2_100_900/",\
            help="filepath to load elites from")
    parser.add_argument("-b", "--body_dim", type=int,\
            help="body dim", \
            default=5)

    args = parser.parse_args()

    list_dir = os.listdir(args.filepath)
    list_dir.sort()

    if "_m3_" in args.filepath:
        my_mode = 3
    elif "_m2_" in args.filepath:
        my_mode = 2
    elif "_m1_" in args.filepath:
        my_mode = 1
    elif "_m0_" in args.filepath:
        my_mode = 0
    else: 
        print("no mode in filepath")

    if "u1_" in args.filepath:
        use_autotomy = 1
    else:
        use_autotomy = 0

    results = {}

    frame_save_path = os.path.join("assets", os.path.split(args.filepath)[-1])

    if os.path.exists(frame_save_path):
        pass
    else:
        os.mkdir(frame_save_path)

    for filename in list_dir:

        if "gen_84" in filename and "npy" in filename:
            
            load_filepath = os.path.join(args.filepath, filename)

            best_agent = 0
            auto_agent = 0
            best_solves = 0

            for agent_idx in range(1):
                my_cmd = f"python -m bevodevo.enjoy -n BackAndForthEnv-v0 -pi MLPBodyPolicy -u {use_autotomy} "\
                        f"-b {args.body_dim} -f {load_filepath} -m {my_mode} -nr 1 -a 1 -i {agent_idx} -e 1"

                print(my_cmd)

                output = subprocess.check_output(my_cmd.split(" "))

                autotomy_used = str(output).count("autotomy used in env? True")
                number_episodes = str(output).count("autotomy used in env?")

                episodes_solved = str(output).count("episode solved? True")
                if number_episodes < 1:
                    
                    import pdb; pdb.set_trace()

                solve_rate = episodes_solved * 1.0 / number_episodes
                auto_rate = autotomy_used * 1.0 / number_episodes

                output_list = str(output).split("\\n")

                print(f"\n {load_filepath}")
                print(f"agent index: {agent_idx}")

                for line in output_list:
                    if "rew:" in line or "autotomy" in line:
                        print("  ", line)

                print(f"       autotomy used in {autotomy_used} of {number_episodes} episodes={auto_rate}")
                print(f"       episode solved in {episodes_solved} of {number_episodes} episodes={solve_rate} \n")

                if solve_rate > best_solves:
                    best_solves = solve_rate
                    best_agent = agent_idx 
                    if auto_rate:
                        auto_agent = agent_idx 

                if auto_rate > 0.0 and auto_agent == -1:
                    auto_agent = agent_idx


            if auto_rate > 0. or my_mode ==0:
                # save frames for later
                if auto_agent != best_agent:
                    tag = "_not_best"
                else:
                    tag = "_best"

                if auto_agent == -1:
                    auto_agent = 0
                    tag = "_no_auto" + tag

                my_cmd_frame = f"python -m bevodevo.enjoy -n BackAndForthEnv-v0 -pi MLPBodyPolicy -u {use_autotomy} "\
                        f"-b {args.body_dim} -f {load_filepath} -m {my_mode} -nr 1 -a 1 -i {auto_agent} -e 1 -s 9"
                os.system(my_cmd_frame)

                frame_list = os.listdir("frames")
                

                for ii in range(2048,-1,-1):
                    if f"frame_agent0_epd0_step{ii:04}.png" in frame_list:
                        last_frame = ii
                        break

                try:
                    print(last_frame)
                except:
                    print("no last frame, expect exception")

                

                frame_save_filename = os.path.join(frame_save_path, os.path.splitext(filename)[0]+f"{tag}.png")
                    
                mv_cmd = f"mv frames/frame_agent0_epd0_step{last_frame:04}.png {frame_save_filename}" 
                print(mv_cmd)
                os.system(mv_cmd)
                os.system("rm frames/*.png")
                        

            # now run again and save a gif at 0.33 scale
            #my_cmd_gif = my_cmd[:-9] + " -e 1 -s 2 -g 0.25"
            my_cmd_gif = f"python -m bevodevo.enjoy -n BackAndForthEnv-v0 -pi MLPBodyPolicy -u {use_autotomy} "\
                    f"-b {args.body_dim} -f {load_filepath} -m {my_mode} -nr 1 -a 1 -i {best_agent} -e 1 -s 2 -g 0.25"
            os.system(my_cmd_gif)








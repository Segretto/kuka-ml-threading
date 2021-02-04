from class_actions_kuka import ActionsThreading
import time
import os
import numpy as np
import time

SAVE_FILE = False

def create_paths():
    path_insertion = '../data/data_insertion/'
    path_backspin = '../data/data_backspin/'
    path_threading = '../data/data_threading/'
    path_general_info = '../data/data_general_info/'
    if not os.path.exists(path_insertion):
        os.makedirs(path_insertion)
    if not os.path.exists(path_backspin):
        os.makedirs(path_backspin)
    if not os.path.exists(path_threading):
        os.makedirs(path_threading)
    if not os.path.exists(path_general_info):
        os.makedirs(path_general_info)
    return path_insertion, path_backspin, path_threading, path_general_info


def count_files(path_threading):
    n = len(os.listdir(path_threading))
    print("THERE ARE " + str(n) + " FILES.")
    return n


def wait_user():
    try:
        while True:
            time.sleep(1)
            pass
    except KeyboardInterrupt:
        print("continuando")


def save_gen_info(file_gen_info, i_bolt):
    MOUNTED = 0
    JAMMED = 1
    NOT_MOUNTED = 2

    outcome = -1
    while outcome != MOUNTED and outcome != JAMMED and outcome != NOT_MOUNTED:
        try:
            outcome = int(input("Insert the outcome for bolt " + str(i_bolt) + ":\n" + "0 - mounted \n1 - jammed \n2 - not mounted\n"))
        except:
            outcome = -1

    with open(file_gen_info, 'a') as f_gen_info:
        f_gen_info.write(str(outcome))

def run_experiment(actions_robot):
    # 0: start the robot
    #    what to log?
    #       Start the timer for batch verification
    actions_robot.robot.open_grip()

    if not SAVE_FILE:
        print("\n\nSAVING FILES TURNED OFF")

    initial_time = time.time()
    n = 16  # number of bolts in the part
    list_bolts = np.arange(n)
    # list_bolts = np.array([15, 0])
    np.random.shuffle(list_bolts)
    angle_backspin = 180  # 180
    angle_threading = 405  # 405

    path_insertion, path_backspin, path_threading, path_general_info = create_paths()
    i_files = count_files(path_threading)

    actions_robot.define_points(px_1=-197.78, py_1=972.55, px_13=-47.70, py_13=971.33)

    for i in range(n):  # starting the experiment
        # what to log? GENERAL INFO
        #       Electrical current axis A6 --> not gonna happen
        #       when it is the case, the exact error that was inserted this run
        print("starting task!")
        print("removing bias.\n\n")
        actions_robot.robot.remove_bias()

        # 1: APPROACH the i-th nut, pick it up, and go to the i-th bolt. Repeat for N times
        print("approaching nut")
        actions_robot.approach_nut(i)
        if i == 0:
            print("waiting user")
            wait_user()
        else:
            print('saving data\n')
            save_gen_info(file_gen_info, list_bolts[i-1])

        print("picking nut")
        actions_robot.pick_nut()
        print("going for the bolt number ", str(list_bolts[i]))
        actions_robot.approach_bolt(list_bolts[i])
        print("PASSEI APPROACH BOLT")

        # 2: INSERTION
        #    what to log?
        #       Force/torques
        #       Position and angles
        nut_start_time = time.time()
        file_insertion = path_insertion + "data_insertion_" + str(i_files).zfill(4) + '_b_' + str(list_bolts[i]).zfill(2) + ".csv"
        if SAVE_FILE:
            actions_robot.robot.start_collect(file_insertion)
        print("PHASE 1: inserting nut")
        actions_robot.print_pos(actions_robot.get_act_position(), 'ponto ' + str(list_bolts[i]))
        actions_robot.insert_nut(fz_desired=-30, wait_time=0)
        actions_robot.robot.sleep(5)
        if SAVE_FILE:
            actions_robot.robot.stop_collect(file_insertion)

        # 3: BACKSPIN
        print("PHASE 2: backspin")
        file_backspin = path_backspin + "data_backspin_" + str(i_files).zfill(4) + '_b_' + str(list_bolts[i]).zfill(2) + ".csv"
        if SAVE_FILE:
            actions_robot.robot.start_collect(file_backspin)
        actions_robot.move_relative(move_a=-angle_backspin, ang_vel=8)  # desejado: 210 graus
        if SAVE_FILE:
            actions_robot.robot.stop_collect(file_backspin)

        # 4: THREADING --> check mounted / check high torque
        print("PHASE 3: threading")
        file_threading = path_threading + "data_threading_" + str(i_files).zfill(4) + '_b_' + str(list_bolts[i]).zfill(2) + ".csv"
        if SAVE_FILE:
            actions_robot.robot.start_collect(file_threading)
        actions_robot.turn_and_torque_check(desired_angle=angle_threading, max_torque=2)  # desejado: 210 + 210 = 420
        if SAVE_FILE:
            actions_robot.robot.stop_collect(file_threading)

        # 5: open gripper, release the nut and go to #1
        print("returning orientation")
        actions_robot.recover_from_threading(angle_recover=(angle_threading-angle_backspin))
        # next action is to approach nut, which is already up there

        # 6: finish operation + stop the timer started at #0 and log it
        # finishing operation and saving general info
        #       Time to finish the assembly of one nut
        #       Which bolt I'm assembling (number i)
        nut_final_time = time.time() - nut_start_time

        if SAVE_FILE:
            file_gen_info = path_general_info + "general_info_" + str(i_files).zfill(4) + '_b_' + str(list_bolts[i]).zfill(2) + ".txt"
            with open(file_gen_info, 'w') as f_gen_info:
                first_row = 'assembly_time, outcome'
                f_gen_info.write(first_row + "\n")
                f_gen_info.write(str(nut_final_time) + ',') # + str(outcome))
            i_files += 1

    print("returning to initial pos")
    actions_robot.approach_nut()

    if SAVE_FILE:
        save_gen_info(file_gen_info, list_bolts[i])

    batch_final_time = time.time() - initial_time # TODO: RE-DO THIS ONE
    # i_batch = 0
    # while os.path.isfile(path_general_info + "batch_" + str(i_batch).zfill(4) + "_time.txt"):
    #     i_batch += 1
    # with open(path_general_info + "batch_" + str(i_batch).zfill(4) + "_time.txt", 'w') as f:
    #     f.write(str(batch_final_time))


if __name__ == '__main__':
    ip = "192.168.10.15"
    port = 6008
    actions_robot = ActionsThreading(ip, port, allow_threading_rotation=False)
    run_experiment(actions_robot)
    print('\nExperiment ended!')
    actions_robot.robot.stop_communication()

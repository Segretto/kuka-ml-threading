from robot_kuka import RobotKuka
import numpy as np
import time
from copy import deepcopy


class ActionsThreading:

    def __init__(self, ip, port, time_loop_ms=40, allow_threading_rotation=True, observation_space=None):

        if observation_space is not None:
            self.observation_space = observation_space
            self.states = np.zeros((self.observation_space, 1))
            self.states_return_alignment = np.zeros((self.observation_space, 1))
            self.states_return = np.zeros((self.observation_space, 1))
            self.states_prev = np.zeros((self.observation_space, 1))
            self.states = np.zeros((self.observation_space, 1))

        self.time_loop_ms = time_loop_ms

        self.bolts_x = np.zeros(16)
        self.bolts_y = np.zeros(16)
        self.nuts_x = np.zeros(16)
        self.nuts_y = np.zeros(16)

        self.robot = self.init_robot(ip, port)
        # self.robot.set_calibration_on()

        self.timer_learning_mean_states = 0.0
        self.timer_learning_mean_states_alignment = 0.0
        self.counter_learning_mean_states = 0.0
        self.states_wrench = np.zeros((6, 1))

        self.init_pos = self.get_act_position()
        self.nut_pick_point = None
        self.point_current_bolt = None
        self.init_pos_point = self.get_act_position()
        self.init_axis = self.robot.get_current_axis_position()

        self.impedance_lin_m = 0
        self.impedance_lin_b = 500
        self.impedance_lin_k = 2500
        self.factor_alignment = 40

        self.impedance_rot_m = 0
        self.impedance_rot_b = 50
        self.impedance_rot_k = 180
        self.factor_threading = 4.5
        self.allow_threading_rotation = allow_threading_rotation

    def my_rounder(self, x, resolution=0.05):
        return round(x / resolution) * resolution

    def init_robot(self, ip, port):
        robot = RobotKuka()
        robot.start_communication(ip, port)
        robot.wait_robot()
        print("Communication running!")
        return robot

    def move_to_point(self, insert_error, point=0, lin_vel=100, ang_vel=6):

        pos_act = self.robot.get_current_position()

        pos_desired = self.get_point_coordinates(point)

        pos_error = self.get_pos_error(insert_error)

        # pos_desired é o calibrado na mão
        move_y = -(pos_desired.x - pos_act.x)
        move_x = -(pos_desired.y - pos_act.y)
        move_z = -(pos_desired.z - pos_act.z)
        move_a = 0
        move_b = 0
        move_c = 0

        print("random position: position = " + str(point) + "\twith error_y = " + str(
            pos_error[1]) + "\twith error_x = " + str(pos_error[0]) + "\twith error_z = " + str(pos_error[2]))

        # primeiro vou pra posição certa
        self.robot.move(x=move_x, y=move_y, z=move_z, a=move_a, b=move_b, c=move_c, li_speed=lin_vel, an_speed=ang_vel)
        self.robot.sleep(2.0)

        pos_act = self.robot.get_current_position()

        # os valores de X e Y estao invertidos
        # corrijo os valores caso estejam fora
        self.fine_adjustment_pos(pos_desired)
        # while abs(pos_desired.x - pos_act.x) > 0.1 or abs(pos_desired.y - pos_act.y) > 0.1 or abs(
        #         pos_desired.z - pos_act.z) > 0.1:
        #     print("ENTREI NA CORRECAO FINA DA POS")
        #     if abs(pos_desired.x - pos_act.x) > 0.1:
        #         # print("error x = ", pos_desired.x - pos_act.x)
        #         if pos_desired.x - pos_act.x > 0:
        #             pos_increment = -0.1
        #         else:
        #             pos_increment = 0.1
        #         self.move_relative(move_y=pos_increment, lin_vel=20)
        #     # else:
        #     if abs(pos_desired.y - pos_act.y) > 0.1:
        #         print("error y = ", pos_desired.y - pos_act.y)
        #         if pos_desired.y - pos_act.y > 0:
        #             pos_increment = -0.1
        #         else:
        #             pos_increment = 0.1
        #         self.move_relative(move_x=pos_increment, lin_vel=20)
        #     # else:
        #     if abs(pos_desired.z - pos_act.z) > 0.1:
        #         # print("error z = ", pos_desired.z - pos_act.z)
        #         if pos_desired.z - pos_act.z > 0:
        #             pos_increment = -0.1
        #         else:
        #             pos_increment = 0.1
        #         self.move_relative(move_z=pos_increment, lin_vel=20)
        #
        #     self.robot.sleep(0.1)
        #     pos_act = self.robot.get_current_position()

        # REMOVING BIAS
        forces = self.robot.get_current_forces_unbiased()
        max_force_bias = 0.5

        if abs(forces.z) > max_force_bias or (forces.x ** 2 + forces.y ** 2) ** 0.5 > max_force_bias:
            print("\nRemoving bias\n")
            self.robot.remove_bias()
            self.robot.sleep(3.0)

        # e agora insiro os erros
        if insert_error:
            print("inserting errors")
            move_y = pos_error[0]
            move_x = pos_error[1]
            move_z = pos_error[2]
            move_a = 0
            move_b = 0
            move_c = 0

            self.robot.move(x=move_x, y=move_y, z=move_z, a=move_a, b=move_b, c=move_c, li_speed=lin_vel,
                            an_speed=ang_vel)
            self.robot.sleep(0.5)

        if pos_error[3] != 0.0 and self.allow_threading_rotation:
            print("turning x = ", pos_error[3])
            move_c = pos_error[3]
            self.robot.move(c=move_c, li_speed=lin_vel, an_speed=ang_vel)
            self.robot.sleep(0.5)

        # print("POSICAO DESEJADA")
        # print("pos_desired.x = ", pos_desired.x)
        # print("pos_desired.y = ", pos_desired.y)
        # print("pos_desired.z = ", pos_desired.z)
        # print("pos_desired.a = ", pos_desired.a)
        # print("pos_desired.b = ", pos_desired.b)
        # print("pos_desired.c = ", pos_desired.c)
        #
        # print("POSICAO ATUAL")
        # print("current_pos.x = ", self.robot.get_current_position().x)
        # print("current_pos.y = ", self.robot.get_current_position().y)
        # print("current_pos.z = ", self.robot.get_current_position().z)
        # print("current_pos.a = ", self.robot.get_current_position().a)
        # print("current_pos.b = ", self.robot.get_current_position().b)
        # print("current_pos.c = ", self.robot.get_current_position().c)

    def get_point_coordinates(self, point):

        # print("POSICAO INICIAL DENTRO FUNC ANTES")
        # print("init_pos.x = ", self.init_pos.x)
        # print("init_pos.y = ", self.init_pos.y)
        # print("init_pos.z = ", self.init_pos.z)
        # print("init_pos.a = ", self.init_pos.a)
        # print("init_pos.b = ", self.init_pos.b)
        # print("init_pos.c = ", self.init_pos.c)
        # print("\n")

        # p1
        # p = [self.get_act_position()] * 4
        # print("QUEM EH P = ", p)
        p_ret = self.get_act_position()
        p_ret.x = self.init_pos.x
        p_ret.y = self.init_pos.y
        p_ret.z = self.init_pos.z
        p_ret.a = self.init_pos.a
        p_ret.b = self.init_pos.b
        p_ret.c = self.init_pos.c

        if point == 0:
            return p_ret
        elif point == 1:
            p_ret.x = self.init_pos.x + 70 + 0.2
            p_ret.y = self.init_pos.y - 1.2
        elif point == 2:
            p_ret.x = self.init_pos.x + 140 + 0.2
            p_ret.y = self.init_pos.y - 2.4
        elif point == 3:
            p_ret.x = self.init_pos.x + 210 + 0.2
            p_ret.y = self.init_pos.y - 3.6

        return p_ret

    def get_pos_error(self, insert_error):
        error_max = 4
        # move_x = (np.random.rand()-0.5)*2*error_max
        # move_y = (np.random.rand()-0.5)*2*error_max
        move_x = np.sign((np.random.rand() - 0.5)) * error_max
        move_y = np.sign((np.random.rand() - 0.5)) * error_max

        move_z = (np.random.rand() - 0.5) * 2 * error_max / 2  # error of maximum +-2 mm
        error_angle = 4
        ang_x = np.random.rand() * error_angle  # in this case, maximum error is 4 degrees

        if not insert_error:
            ang_x = 0.0
            move_x = 0.0
            move_y = 0.0
            move_z = 0.0
            # position = 0.0

        return [move_x, move_y, move_z, ang_x]

    def move_relative(self, move_x=0.0, move_y=0.0, move_z=0.0, move_a=0.0, move_b=0.0, move_c=0.0, lin_vel=100,
                      ang_vel=6):
        self.robot.move(x=move_x, y=move_y, z=move_z, a=move_a, b=move_b, c=move_c, li_speed=lin_vel, an_speed=ang_vel)
        self.robot.sleep(0.1)

    def rotate_desired_angle(self, move_a=0.0, move_b=0.0, move_c=0.0, ang_vel=6):
        self.robot.move(a=move_a, b=move_b, c=move_c, an_speed=ang_vel)
        self.robot.sleep(0.1)

    def contact(self, fx_desired=0.0, fy_desired=0.0, fz_desired=0.0, mx_desired=0.0,
                my_desired=0.0, mz_desired=0.0, wait_time=0.0):

        self.turn_impedance_on(fx_desired, fy_desired, fz_desired, mx_desired, my_desired, mz_desired)

        if wait_time == 0.0:

            while abs(self.robot.get_current_forces_unbiased().z) < 0.9 * abs(fz_desired):
                # print(self.robot.get_current_forces_unbiased().z)
                # print("\n\nFZ = ", self.robot.get_current_forces_unbiased().z)
                continue
        else:
            time_start = self.time_ms()
            wait_time = wait_time * 1000
            while (self.time_ms() - time_start) < wait_time:
                # continue
                f = self.robot.get_current_forces_unbiased()
                # Fxy = math.sqrt(f.x**2 + f.y**2)
                # print("Fxy = ", Fxy)
                # self.time_ms() - time_starts
                # print("\nFZ = ", f.z)
                # print("DELTA_T = ", self.time_ms() - time_start)

    def turn_impedance_on(self, fx_desired=0.0, fy_desired=0.0, fz_desired=0.0, mx_desired=0.0, my_desired=0.0,
                          mz_desired=0.0):
        m = self.impedance_lin_m
        b = self.impedance_lin_b
        k = self.impedance_lin_k
        m_ang = self.impedance_rot_m
        b_ang = self.impedance_rot_b
        k_ang = self.impedance_rot_k
        self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=fx_desired, mode='d')
        self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=fy_desired, mode='d')
        self.robot.activate_impedance_control_z(m=m, b=b, k=k, ref=fz_desired, mode='d')
        if self.allow_threading_rotation:
            self.robot.activate_impedance_control_c(m=m_ang, b=b_ang, k=k_ang, ref=mx_desired, mode='d')
            self.robot.activate_impedance_control_b(m=m_ang, b=b_ang, k=k_ang, ref=my_desired, mode='d')
        self.robot.activate_impedance_control_a(m=m_ang, b=b_ang, k=k_ang, ref=mz_desired, mode='d')

    def turn_impedance_off(self, fx_desired=0.0, fy_desired=0.0, fz_desired=0.0, mx_desired=0.0, my_desired=0.0,
                           mz_desired=0.0):
        m = self.impedance_lin_m
        b = self.impedance_lin_b
        k = self.impedance_lin_k
        m_ang = self.impedance_rot_m
        b_ang = self.impedance_rot_b
        k_ang = self.impedance_rot_k
        self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=fx_desired, mode='d')
        self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=fy_desired, mode='d')
        self.robot.activate_impedance_control_z(m=m, b=b, k=k, ref=fz_desired, mode='d')
        if self.allow_threading_rotation:
            self.robot.activate_impedance_control_c(m=m_ang, b=b_ang, k=k_ang, ref=mx_desired, mode='d')
            self.robot.activate_impedance_control_b(m=m_ang, b=b_ang, k=k_ang, ref=my_desired, mode='d')
        self.robot.activate_impedance_control_a(m=m_ang, b=b_ang, k=k_ang, ref=mz_desired, mode='d')

    def thread_now(self):
        # delta_angle = 190 + (180 - abs(self.robot.get_current_position().a))
        # print("\n\nTENHO QUE ROSQUEAR DE = ", delta_angle)

        threading_angle = 90

        # TODO: verificar sinal
        # while (self.robot.get_current_position().a - desired_angle) > 0.0:
        # print("acao controle = ", delta_angle)
        self.rotate_desired_angle(move_a=threading_angle, ang_vel=20)
        #   self.robot.sleep(0.1)
        # self.rotate_desired_angle(move_a = desired_angle, ang_vel = 16)

    def time_ms(self):
        return time.time() * 1000

    def get_init_pos(self):
        return self.robot.get_current_position()

    def get_act_position(self):
        return self.robot.get_current_position()

    def states_learning(self, get_time=False):
        # this method verifies every time it is ticked, if 40 ms has been elapsed since last loop. If not,
        # it keeps increasing self.states (states = [fx, fy, fz, mx, my, mz, theta_z]) by adding to it the current
        # values of the desired variables and incrementing a counter, while returning the last value. If time
        # elapsed is indeed more than 40 ms, then it gets the average of the last values and flushes the new value.
        time_now = self.time_ms() - self.timer_learning_mean_states
        # print("time_now nos states = ", time_now)

        # ######states = [0, 0, fz, 0, 0, mz, theta_z]
        if time_now < self.time_loop_ms:
            self.states_wrench = self.robot.get_current_forces_unbiased()
            # print("Fx = ", str(self.states_wrench.x))
            # print("Fy = ", str(self.states_wrench.y))
            # print()
            self.states[0] = self.states[0] + self.states_wrench.x
            self.states[1] = self.states[1] + self.states_wrench.y
            self.states[2] = self.states[2] + self.states_wrench.z
            self.states[3] = self.states[3] + self.states_wrench.a
            self.states[4] = self.states[4] + self.states_wrench.b
            self.states[5] = self.states[5] + self.states_wrench.c
            self.counter_learning_mean_states += 1
        else:
            if self.counter_learning_mean_states == 0.0:
                self.states_return = np.zeros((self.observation_space, 1))
            else:
                self.states_return = np.divide(self.states, self.counter_learning_mean_states)
            # print("Qntde de elementos no vetor de estados = ", self.counter_learning_mean_states)
            # self.states_return[6] = theta_d - theta_act
            # self.states_return[2] = self.my_rounder(x = self.states_return[2], resolution = 0.25)
            current_pos = self.robot.get_current_position()
            self.states_return[6] = current_pos.x - self.init_pos_point.x
            self.states_return[7] = current_pos.y - self.init_pos_point.y
            self.states_return[8] = current_pos.z - self.init_pos_point.z
            self.states_return[9] = current_pos.a - self.init_pos_point.a
            self.states_return[10] = current_pos.b - self.init_pos_point.b
            self.states_return[11] = current_pos.c - self.init_pos_point.c
            # self.states_return[12] = self.feedback_msg_adm.vel_cart[0]
            # self.states_return[13] = self.feedback_msg_adm.vel_cart[1]
            # self.states_return[14] = self.feedback_msg_adm.vel_cart[2]
            # self.states_return[15] = self.feedback_msg_adm.vel_cart[3]
            # self.states_return[16] = self.feedback_msg_adm.vel_cart[4]
            # self.states_return[17] = self.feedback_msg_adm.vel_cart[5]
            self.states = np.zeros((self.observation_space, 1))
            self.counter_learning_mean_states = 0.0
            # self.states_return = np.append(self.states_wrench, self.get_act_rotation_rounded())
            self.timer_learning_mean_states = self.time_ms()
            # print(self.timer_learning_mean_states)

        # # states = [theta_d - theta_z]
        # if (time_now < self.time_loop_ms):
        # 	# self.states_return = theta_d - self.get_act_rotation_rounded()
        # 	self.states_return = theta_d - theta_act
        # else:
        # 	self.timer_learning_mean_states = self.time_ms()

        if get_time:
            return time_now

        if self.observation_space == 1:
            # 	self.states_return.shape = (1,1,self.observation_space)
            # else:
            self.states_return = np.asarray([[[self.states_return]]])

        self.states_return.shape = (1, 1, self.observation_space)  # RNN
        # self.states_return.shape = (1, self.observation_space) # MLP

        # print("states = ", self.states_return)
        # print("\n")
        # print(self.states_return)
        return self.states_return

    def get_time_alignment(self, get_time=False):
        return self.states_learning(get_time=True)

    def get_time(self, get_time=False):
        return self.states_learning(get_time=True)

    def take_action_alignment(self, action):
        f_desired = 10
        m = self.impedance_lin_m / self.factor_alignment
        b = self.impedance_lin_b / self.factor_alignment
        k = self.impedance_lin_k / self.factor_alignment

        self.robot.activate_impedance_control_z(m=m, b=(self.factor_alignment / 2) * b,
                                                k=(self.factor_alignment / 2) * k, ref=-2 * f_desired, mode='d')

        if action == 0:
            self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=0.0, mode='d')
            self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=f_desired, mode='d')
        elif action == 1:
            self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=0.0, mode='d')
            self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=-f_desired, mode='d')
        elif action == 2:
            self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=0.0, mode='d')
            self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=f_desired, mode='d')
        elif action == 3:
            self.robot.activate_impedance_control_x(m=m, b=b, k=k, ref=0.0, mode='d')
            self.robot.activate_impedance_control_y(m=m, b=b, k=k, ref=-f_desired, mode='d')

    def take_action(self, action):
        tx_desired = 8.5
        tz_desired = 20
        m = self.impedance_rot_m / self.factor_threading
        b = self.impedance_rot_b / self.factor_threading
        k = self.impedance_rot_k / self.factor_threading

        if self.allow_threading_rotation:
            if action == 0:
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=0.0, mode='d')
                self.robot.activate_impedance_control_c(m=m, b=b, k=k, ref=tx_desired, mode='d')
            elif action == 1:
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=0.0, mode='d')
                self.robot.activate_impedance_control_c(m=m, b=b, k=k, ref=-tx_desired, mode='d')
            elif action == 2:
                self.robot.activate_impedance_control_c(m=m, b=b, k=k, ref=0.0, mode='d')
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=tz_desired, mode='d')
            elif action == 3:
                self.robot.activate_impedance_control_c(m=m, b=b, k=k, ref=0.0, mode='d')
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=-tz_desired, mode='d')
        else:
            # print("Action = ", action)
            if action == 0:
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=tz_desired, mode='d')
            elif action == 1:
                self.robot.activate_impedance_control_a(m=m, b=b, k=k, ref=-tz_desired, mode='d')

    def turn_x_initial_orientation(self):

        print("Meu C = ", self.robot.get_current_position().c)

        theta_u = 0.0

        if self.robot.get_current_position().c > 0:
            theta_u = 180 - self.robot.get_current_position().c

        if self.robot.get_current_position().c < 0:
            theta_u = -180 - self.robot.get_current_position().c

        self.rotate_desired_angle(move_c=theta_u, ang_vel=20)

        # ajuste fino
        # print("error = ", abs(180) - abs(self.robot.get_current_position().c))
        # while abs(180) - abs(self.robot.get_current_position().c) > 0.1:
        #     # print("returning c")
        # #     print("init pos A = ", self.init_pos.a)
        #     print("error C = ", abs(180) - abs(self.robot.get_current_position().c))
        #     if self.robot.get_current_position().c > 0:
        #         angle_increment = -0.05
        #     else:
        #         angle_increment = 0.05
        #     self.rotate_desired_angle(move_c = angle_increment, ang_vel = 20)

    def turn_y_initial_orientation(self):

        # while abs(self.robot.get_current_position().b - self.init_pos.b) > 0.2:
        #     print("returning y")
        #     if self.init_pos.b - self.robot.get_current_position().b > 0:
        #         angle_increment = -0.1
        #     else:
        #         angle_increment = 0.1
        #     self.rotate_desired_angle(move_b = angle_increment, ang_vel = 20)
        #     self.robot.sleep(0.01)
        print("Meu B = ", self.robot.get_current_position().b)

        # if self.robot.get_current_position().b > 0:
        #     theta_u = 0 - self.robot.get_current_position().b
        #
        # if self.robot.get_current_position().b < 0:
        if self.robot.get_current_position().b > 0:
            theta_u = self.robot.get_current_position().b

        if self.robot.get_current_position().b < 0:
            theta_u = - self.robot.get_current_position().b

        self.rotate_desired_angle(move_b=theta_u, ang_vel=20)

        # ajuste fino
        # print("error = ", abs(0) - abs(self.robot.get_current_position().b))
        # while abs(0) - abs(self.robot.get_current_position().b) > 0.1:
        #     # print("returning b")
        # #     print("init pos A = ", self.init_pos.a)
        #     print("error B = ", abs(0) - abs(self.robot.get_current_position().b))
        #     if self.robot.get_current_position().b > 0:
        #         angle_increment = -0.05
        #     else:
        #         angle_increment = 0.05
        #     self.rotate_desired_angle(move_b = angle_increment, ang_vel = 20)

    def turn_z_initial_orientation(self):

        current_a6 = self.robot.get_current_axis_position().a6

        current_a = self.robot.get_current_position().a

        # # 1o quadrante negativo
        # if current_a6 < 0:
        #     if current_a < 0:
        #         if current_a < -90:
        #             self.rotate_desired_angle(move_a = -90, ang_vel = 15)
        #         elif current_a < 0:
        #             self.rotate_desired_angle(move_a = -180, ang_vel = 15)
        #     elif current_a > 0:
        #         self.rotate_desired_angle(move_a = -270, ang_vel = 15)
        # else:
        #
        #
        #
        # delta_angle = -(90 + current_a)

        # # print("returning z")
        # # print("Meu theta = ", self.robot.get_current_position().a)
        #
        # current_a = self.robot.get_current_position().a
        #
        # # if current_a > 0:
        # #     theta_u = -(180 - current_a + 90)
        # #
        # # if current_a < 0:
        # #     if current_a > -90:
        # #         theta_u = 90 - current_a
        # #     else:
        # #         theta_u = current_a - 90
        #
        # # self.rotate_desired_angle(move_a = theta_u, ang_vel = 20)
        # # self.rotate_desired_angle(move_a = -210, ang_vel = 20)
        #
        # if current_a < 0:
        #     delta_angle = -60 + abs(current_a)
        #     self.rotate_desired_angle(move_a = delta_angle, ang_vel = 20)
        #
        # current_a = self.robot.get_current_position().a
        # delta_angle = -abs(self.init_pos.a - (180 - current_a))
        # self.rotate_desired_angle(move_a = delta_angle, ang_vel = 20)
        #
        #
        #
        # # ajuste fino
        # # print("error = ", abs(-90) - abs(self.robot.get_current_position().a))
        # # while abs(-90) - abs(self.robot.get_current_position().a) > 0.1:
        # #     # print("returning z")
        # # #     print("init pos A = ", self.init_pos.a)
        # #     print("error A = ", abs(-90) - abs(self.robot.get_current_position().a))
        # #     if -90 - self.robot.get_current_position().a > 0:
        # #         angle_increment = -0.05
        # #     else:
        # #         angle_increment = 0.05
        # #     self.rotate_desired_angle(move_a = angle_increment, ang_vel = 20)

    def fine_adjustment_angle(self, pos_desired):

        # while (abs(-90) - abs(self.robot.get_current_position().a) > 0.1) or (abs(self.robot.get_current_position()
        # .b) > 0.1) or (abs(180) - abs(self.robot.get_current_position().c) > 0.1):
        current_pos = self.robot.get_current_position()
        while (abs(pos_desired.a) - abs(current_pos.a) > 0.1) or \
                (abs(pos_desired.b) - abs(current_pos.b) > 0.1) or \
                (abs(pos_desired.c) - abs(current_pos.c) > 0.1):
            # print("returning z")
            #     print("init pos A = ", self.init_pos.a)
            if abs(pos_desired.a) - abs(current_pos.a) > 0.1:
                print("error A = ", abs(pos_desired.a) - abs(current_pos.a))

                if abs(pos_desired.a) - abs(current_pos.a) > 1:  # when there is too much error
                    increment = 1
                else:
                    increment = 0

                if pos_desired.a - current_pos.a > 0:
                    angle_increment = -0.05 - increment
                else:
                    angle_increment = 0.05 + increment
                self.rotate_desired_angle(move_a=angle_increment, ang_vel=20)
            else:
                if abs(pos_desired.b) - abs(current_pos.b) > 0.1:
                    print("error B = ", abs(pos_desired.b) - abs(current_pos.b))
                    if current_pos.b > 0:
                        angle_increment = 0.05
                    else:
                        angle_increment = -0.05
                    self.rotate_desired_angle(move_b=angle_increment, ang_vel=20)
                else:
                    if abs(pos_desired.c) - abs(current_pos.c) > 0.1:
                        print("error C = ", abs(pos_desired.c) - abs(current_pos.c))
                        if current_pos.c > 0:
                            angle_increment = 0.05
                        else:
                            angle_increment = -0.05
                        self.rotate_desired_angle(move_c=angle_increment, ang_vel=20)
            self.robot.sleep(0.1)
            current_pos = self.robot.get_current_position()

    def fine_adjustment_pos(self, pos_desired):

        pos_act = self.get_act_position()

        while abs(pos_desired.x - pos_act.x) > 0.1 or abs(pos_desired.y - pos_act.y) > 0.1 or \
                abs(pos_desired.z - pos_act.z) > 0.1:

            error_x = pos_desired.x - pos_act.x
            error_y = pos_desired.y - pos_act.y
            error_z = pos_desired.z - pos_act.z

            # print("x = ", error_x)
            # print("y = ", error_y)
            # print("z = ", error_z)

            print("\n\nONDE EU QUERO CHEGAR")
            pos_desired.plot()
            print("\nONDE EU ESTOU")
            self.get_act_position().plot()

            # print("ENTREI NA CORRECAO FINA DA POS")
            if abs(error_x) > 0.1:

                if abs(error_x) > 1:  # when there is too much error
                    increment = abs(error_x)
                else:
                    increment = 0

                if pos_desired.x - pos_act.x > 0:
                    pos_increment = -0.1 - increment
                else:
                    pos_increment = 0.1 + increment
                self.move_relative(move_y=pos_increment, lin_vel=20)

            if abs(error_y) > 0.1:

                if abs(error_y) > 1:  # when there is too much error
                    increment = abs(error_y)
                else:
                    increment = 0

                if pos_desired.y - pos_act.y > 0:
                    pos_increment = -0.1 - increment
                else:
                    pos_increment = 0.1 + increment
                self.move_relative(move_x=pos_increment, lin_vel=20)

            if abs(error_z) > 0.1:

                if abs(error_z) > 1:  # when there is too much error
                    increment = 1
                else:
                    increment = 0

                if pos_desired.z - pos_act.z > 0:
                    pos_increment = -0.1 - increment
                else:
                    pos_increment = 0.1 + increment
                self.move_relative(move_z=pos_increment, lin_vel=20)

            self.robot.sleep(0.1)
            # pos_act = self.robot.get_current_position()
            # self.print_pos(pos_desired, "desejado")
            # self.print_pos(pos_act, "atual")

    def turn_z_desired_angle(self, desired_angle=0.0):
        epsilon = 0.05
        A_error = desired_angle - self.robot.get_current_position().a

        if A_error > 0:
            angle_increment = 0.2
        else:
            angle_increment = -0.2

        while abs(desired_angle - self.robot.get_current_position().a) > epsilon:
            self.rotate_desired_angle(move_a=angle_increment)
            self.robot.sleep(0.1)

    def reward_learning_threading(self, force_desired=20.0):

        # if np.ndim(self.states_return) < 3:
        #     self.states_return = np.asarray([[[self.states_return]]])
        # print(self.states_return)
        self.states_return.shape = (1, 1, self.observation_space)

        reward_fxy = np.sqrt(self.states_return[0][0][0] ** 2 + self.states_return[0][0][1] ** 2) / 30
        reward_mxy = np.sqrt(self.states_return[0][0][3] ** 2 + self.states_return[0][0][4] ** 2) / 2
        reward_mz = abs(self.states_return[0][0][5]) / 2
        # + self.states_return[0][0][5]**2)
        reward_fz = abs(force_desired - self.states_return[0][0][2]) / abs(force_desired)

        # print("reward_fxy = ", reward_fxy)
        # print("reward_fz = ", reward_fz)
        # print("reward_mxy = ", reward_mxy)
        # print("reward_mz = ", reward_mz)
        # print("total_esforcos = ", reward)

        reward = -(reward_fxy + reward_mxy + reward_fz + reward_mz)

        return reward_fxy + reward_mxy + reward_fz + reward_mz

    # def return_init_orientation(self):
    #     self.turn_x_initial_orientation()
    #     self.turn_z_initial_orientation()

    def approach_nut(self, i):
        self.robot.deactivate_impedance_control()

        if self.nut_pick_point is None:
            # saving nut position, because I will need to go back there
            self.nut_pick_point = self.get_act_position()
        else:
            current_pos = self.get_act_position()

            # positioning on the top of the nut
            nut_pick_point_x = self.nut_pick_point.x + self.nuts_x[i]
            nut_pick_point_y = self.nut_pick_point.y + self.nuts_y[i]

            y = -(nut_pick_point_x - current_pos.x)
            x = -(nut_pick_point_y - current_pos.y)
            z = -(self.nut_pick_point.z - current_pos.z)
            a = abs(self.nut_pick_point.a) - abs(current_pos.a)
            b = self.nut_pick_point.b - current_pos.b
            c = self.nut_pick_point.c - current_pos.c
            self.move_relative(move_x=x, move_y=y, move_z=z, move_a=a)  # , move_b=b, move_c=c, lin_vel=100)

            print("fine adjustment pos")
            self.fine_adjustment_pos(self.nut_pick_point)
            print("fine adjustment angle")
            self.fine_adjustment_angle(self.nut_pick_point)

    def pick_nut(self):
        self.robot.open_grip()

        # when I'm already on the top of the nut
        z = 24
        self.move_relative(move_z=z, lin_vel=5)
        self.robot.sleep(1)

        self.robot.close_grip()
        self.robot.sleep(1)

        self.move_relative(move_z=-z, lin_vel=30)
        self.robot.sleep(2)

    def calculate_theta(self, px_1, py_1, px_13, py_13):
        delta_x = px_13 - px_1
        delta_y = py_13 - py_1
        return np.arctan(delta_y/delta_x)

    def define_points(self, px_1, py_1, px_13, py_13):
        # BOLTS
        theta = -self.calculate_theta(px_1, py_1, px_13, py_13)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        # calculating without transformation
        delta_bolts = 50
        delta_x_nuts = 33.857
        delta_y_nuts = 80
        for i in range(4):
            self.bolts_x[i] = delta_bolts * i
            self.bolts_x[i + 4] = delta_bolts * i
            self.bolts_x[i + 8] = delta_bolts * i
            self.bolts_x[i + 12] = delta_bolts * i

            self.bolts_y[4 * i] = delta_bolts * i
            self.bolts_y[4 * i + 1] = delta_bolts * i
            self.bolts_y[4 * i + 2] = delta_bolts * i
            self.bolts_y[4 * i + 3] = delta_bolts * i

        # transforming with the rotation
        for i in range(16):
            self.bolts_x[i] = R[0][0] * self.bolts_x[i] + R[0][1] * self.bolts_y[i]
            self.bolts_y[i] = R[1][0] * self.bolts_x[i] + R[1][1] * self.bolts_y[i]

            # NUTS
            if i % 2 == 0:
                self.nuts_y[i] = 0
                self.nuts_x[i] = delta_x_nuts*(i/2)  # each delta is added for each iterator i
            else:
                self.nuts_y[i] = delta_y_nuts
                self.nuts_x[i] = delta_x_nuts*((i-1)/2)  # same spacing as the even ones

    def approach_bolt(self, i):
        # when getting from the bench on tool's frame
        # y_first_bolt = 42.3
        # x_first_bolt = 122.3

        print("\nPOSICAO ATUAL")
        self.get_act_position().plot()

        # when getting from the board on tool's frame
        x_first_bolt = -68
        y_first_bolt = 445

        pos_bolt = deepcopy(self.nut_pick_point)
        pos_bolt.x = pos_bolt.x - self.bolts_x[i] + x_first_bolt
        pos_bolt.y = pos_bolt.x - self.bolts_y[i] + y_first_bolt

        print("\nPOSICAO DESEJADA")
        pos_bolt.plot()
        print("\nPOSICAO ATUAL")
        self.get_act_position().plot()

        # this is for the bolts when I already know the first one
        self.move_relative(move_x=-self.bolts_x[i] + x_first_bolt, move_y=-self.bolts_y[i] + y_first_bolt)
        self.robot.sleep(3)
        print("PASSEI MOVE RELATIVE")
        # print("\nentrei na posicao de ajuste fino do parafuso\n")
        self.fine_adjustment_pos(pos_desired=pos_bolt)
        print("PASSEI O FINE ADJUST")
        # self.point_current_bolt = self.get_act_position()
        # self.move_relative(move_z=8, lin_vel=10)  # This is just to make everything faster.
        # self.robot.sleep(1)                       # Previous point saved is to avoid collision.

    def insert_nut(self, fz_desired=0, wait_time=0):
        self.contact(fz_desired=fz_desired, wait_time=wait_time)

    def recover_from_threading(self, angle_recover):
        # 0: shutdown impedance controller
        self.turn_impedance_off()
        # 1: open gripper
        self.robot.open_grip()
        # 2: wait
        self.robot.sleep(1)
        # 3: move up
        current_pos = self.get_act_position()
        z_move_up = current_pos.z - self.point_current_bolt.z  # point_current_bolt.z is the initial height
        self.move_relative(move_z=z_move_up, lin_vel=20)
        self.robot.sleep(1)
        # 4: rotate to initial angle
        self.move_relative(move_a=-angle_recover, ang_vel=24)
        self.robot.sleep(3)
        # 4.1: guarantee initial orientation
        self.fine_adjustment_angle(self.nut_pick_point)

    def turn_and_torque_check(self, desired_angle, max_torque):
        counter = 0
        previous_angle = None

        self.turn_impedance_on(fz_desired=-30, mz_desired=-15.5)

        # test with sending position and monitoring torque

        while abs(counter) < desired_angle and abs(self.robot.get_current_forces_unbiased().a) < max_torque:
            current_angle = self.get_act_position().a

            if previous_angle is None:
                previous_angle = current_angle

            delta_angle = current_angle - previous_angle

            if abs(delta_angle) < 5:
                counter += current_angle - previous_angle
            previous_angle = current_angle

            self.robot.sleep(0.3)

    def print_pos(self, pos, text):
        # print(text, "\nx = ", pos.x, "\ny = ", pos.y, "\nz = ", pos.z, "\na = ",
        #       pos.a, "\nb = ", pos.b, "\nc = ", pos.c, "\n\n")
        print(text)
        pos.plot()

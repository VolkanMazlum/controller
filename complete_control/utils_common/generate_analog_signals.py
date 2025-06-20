import numpy as np
from config.core_models import ExperimentParams, SimulationParams
from plant.robot1j import Robot1J


def generate_signals(exp: ExperimentParams, sim: SimulationParams):
    """Generate trajectory and motor commands for the simulation.

    Returns:
        tuple: (trajectory, motor_commands) arrays for the simulation
    """

    res = sim.resolution
    n_trials = sim.n_trials
    time_sim = sim.time_move
    time_prep = sim.time_prep
    time_post = sim.time_post
    total_time_trial = time_prep + time_sim + time_post
    total_time = total_time_trial * n_trials

    time_vect = np.linspace(
        0, total_time_trial, num=int(np.round(total_time_trial / res)), endpoint=True
    )
    time_sim_vec = np.linspace(
        0, time_sim, num=int(np.round(time_sim / res)), endpoint=True
    )
    total_time_vec = np.linspace(
        0, total_time, num=int(np.round(total_time / res)), endpoint=True
    )

    dynSys = Robot1J(robot=exp.robot_spec)
    njt = dynSys.numVariables()

    # Joint space
    init_pos = np.array([exp.init_pos_angle_rad])
    tgt_pos = np.array([exp.tgt_pos_angle_rad])

    ## Compute minimum jerk trajectory (input to Planner)
    def minimumJerk(x_init, x_des, timespan):
        T_max = timespan[len(timespan) - 1]
        tmspn = timespan.reshape(timespan.size, 1)

        a = 6 * (x_des - x_init) / np.power(T_max, 5)
        b = -15 * (x_des - x_init) / np.power(T_max, 4)
        c = 10 * (x_des - x_init) / np.power(T_max, 3)
        d = np.zeros(x_init.shape)
        e = np.zeros(x_init.shape)
        g = x_init

        pol = np.array([a, b, c, d, e, g])
        pp = (
            a * np.power(tmspn, 5) + b * np.power(tmspn, 4) + c * np.power(tmspn, 3) + g
        )

        return pp, pol

    trj, pol = minimumJerk(init_pos, tgt_pos, time_sim_vec)  # Joint space (angle)
    trj_ee = dynSys.forwardKin(trj)  # End-effector space

    trj_prep = trj[0] * np.ones(int(time_prep / res))
    trj_post = 0 * np.ones(int(time_post / res))
    trj = np.tile(np.concatenate((trj_prep, trj.flatten(), trj_post)), n_trials)

    ## Compute motor commands (input to Motor Cortex)
    # Double derivative of the trajectory
    def minimumJerk_ddt(x_init, x_des, timespan):
        T_max = timespan[len(timespan) - 1]
        tmspn = timespan.reshape(timespan.size, 1)

        a = 120 * (x_des - x_init) / np.power(T_max, 5)
        b = -180 * (x_des - x_init) / np.power(T_max, 4)
        c = 60 * (x_des - x_init) / np.power(T_max, 3)
        d = np.zeros(x_init.shape)

        pol = np.array([a, b, c, d])
        pp = (
            a * np.power(tmspn, 3) + b * np.power(tmspn, 2) + c * np.power(tmspn, 1) + d
        )
        ("pp: ", len(pp))
        return pp, pol

    # Time and value of the minimum jerk curve
    def minJerk_ddt_minmax(x_init, x_des, timespan):

        T_max = timespan[len(timespan) - 1]
        t1 = T_max / 2 - T_max / 720 * np.sqrt(43200)
        t2 = T_max / 2 + T_max / 720 * np.sqrt(43200)
        pp, pol = minimumJerk_ddt(x_init, x_des, timespan)

        ext = np.empty(shape=(2, x_init.size))
        ext[:] = 0.0
        t = np.empty(shape=(2, x_init.size))
        t[:] = 0.0

        for i in range(x_init.size):
            if x_init[i] != x_des[i]:
                tmp = np.polyval(pol[:, i], [t1, t2])
                ext[:, i] = np.reshape(tmp, (1, 2))
                t[:, i] = np.reshape([t1, t2], (1, 2))

        return t, ext

    # Compute the torques via inverse dynamics
    def generateMotorCommands(init_pos, des_pos, time_vector):
        # Last simulation time
        T_max = time_vector[len(time_vector) - 1]
        # Time and value of the minimum jerk curve
        ext_t, ext_val = minJerk_ddt_minmax(init_pos, des_pos, time_vector)

        # Approximate with sin function
        tmp_ext = np.reshape(ext_val[0, :], (1, njt))  # First extreme (positive)
        tmp_sin = np.sin((2 * np.pi * time_vector / T_max))
        mp_sin = np.reshape(tmp_sin, (tmp_sin.size, 1))

        # Motor commands: Inverse dynamics given approximated acceleration
        dt = (time_vector[1] - time_vector[0]) / 1e3
        pos, pol = minimumJerk(init_pos, des_pos, time_vector)
        vel = np.gradient(pos, dt, axis=0)
        acc = tmp_ext * tmp_sin
        mcmd = dynSys.inverseDyn(pos, vel, acc)
        return mcmd[0]

    motorCommands = generateMotorCommands(init_pos, tgt_pos, time_sim_vec / 1e3)
    mc_prep = 0 * np.ones(int(time_prep / res))
    mc_post = 0 * np.ones(int(time_post / res))
    motorCommands = np.tile(
        np.concatenate((mc_prep, motorCommands.flatten(), mc_post)), n_trials
    )

    return trj, motorCommands


if __name__ == "__main__":
    trj, motorCommands = generate_signals()
    sim = SimulationParams()
    print(f"Generated trajectory shape: {trj.shape}")
    print(trj)
    print(f"Generated motor commands shape: {motorCommands.shape}")
    print(motorCommands)
    print(
        f"expected length: {(sim.time_prep + sim.time_move + sim.time_post)*sim.n_trials/sim.resolution}"
    )

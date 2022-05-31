
from arm_1dof.bullet_arm_1dof import BulletArm1Dof
from arm_1dof.robot_arm_1dof import RobotArm1Dof
import numpy as np
import time

data_cmd = "./complete_control/data/inputCmd_tot.csv"

inputCmd_tot = np.loadtxt(data_cmd)
######################## INIT BULLET ##########################

# Initialize bullet and load robot
bullet = BulletArm1Dof()
# bullet.InitPybullet()

import pybullet as p
bullet.InitPybullet(bullet_connect=p.GUI)#, g=[0.0, 0.0 , -9.81])
bullet_robot = bullet.LoadRobot()


upperarm = p.getLinkState(bullet_robot._body_id, RobotArm1Dof.UPPER_ARM_LINK_ID,
    computeLinkVelocity=True)[0]
forearm = p.getLinkState(bullet_robot._body_id, RobotArm1Dof.FOREARM_LINK_ID,
            computeLinkVelocity=True)[0]
hand = p.getLinkState(bullet_robot._body_id, RobotArm1Dof.HAND_LINK_ID,
            computeLinkVelocity=True)[0]
# hand = bullet_robot.EEPose()[0]
_init_pos = [hand[0], hand[2]]
rho = np.linalg.norm(np.array(upperarm)-np.array(hand))
z = upperarm[2] -rho*np.cos(1.57)
y = upperarm[0] +rho*np.sin(1.57)
_tgt_pos  = [y,z]


res = 0.0001
step    = 0 # simulation step

for step in range(len(inputCmd_tot)):
     # Total torques

    # Set joint torque
    bullet_robot.SetJointTorques(joint_ids=[RobotArm1Dof.ELBOW_JOINT_ID], torques=[inputCmd_tot[step]] )

    # Integrate dynamical system
    bullet.Simulate(sim_time=res)
    time.sleep(0.0004)

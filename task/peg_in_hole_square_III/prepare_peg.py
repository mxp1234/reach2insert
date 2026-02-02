#!/usr/bin/env python3
"""
Peg准备脚本

功能：
1. 移动机械臂到安全位置
2. 张开夹爪
3. 等待3秒（用户放置peg）
4. 闭合夹爪

Usage:
    python prepare_peg.py
"""

import sys
import os
import time
import numpy as np
import requests

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'serl_robot_infra'))

from franka_env.utils.rotations import euler_2_quat


class PegPreparer:
    def __init__(self):
        from task.peg_in_hole_square_III.config import EnvConfig
        self.config = EnvConfig()
        self.url = self.config.SERVER_URL
        self.hz = 10

        # 安全位置 - 较高的位置，方便放置peg
        self.safe_pose = np.array([0.5, 0.0, 0.35, np.pi, 0, 0])

    def _recover(self):
        """清除错误"""
        requests.post(self.url + "clearerr")

    def _get_state(self):
        """获取机器人状态"""
        ps = requests.post(self.url + "getstate").json()
        return np.array(ps["pose"])

    def _send_pos_command(self, pos):
        """发送位置命令"""
        self._recover()
        arr = np.array(pos).astype(np.float32)
        requests.post(self.url + "pose", json={"arr": arr.tolist()})

    def _open_gripper(self):
        """打开夹爪"""
        requests.post(self.url + "open_gripper")

    def _close_gripper(self):
        """闭合夹爪"""
        requests.post(self.url + "close_gripper")

    def _interpolate_move(self, goal_euler, timeout=2.0):
        """插值移动到目标位置"""
        goal_quat = np.concatenate([goal_euler[:3], euler_2_quat(goal_euler[3:])])
        currpos = self._get_state()
        steps = int(timeout * self.hz)
        path = np.linspace(currpos, goal_quat, steps)

        for p in path:
            self._send_pos_command(p)
            time.sleep(1 / self.hz)

    def run(self):
        """执行准备流程"""
        print("\n" + "=" * 50)
        print("Peg 准备脚本")
        print("=" * 50)

        # 步骤1: 设置精确模式
        print("\n[1/5] 设置精确模式...")
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)
        time.sleep(0.5)
        self._recover()

                # 步骤3: 打开夹爪
        print("\n[3/5] 打开夹爪...")
        self._open_gripper()
        time.sleep(0.5)

        # 步骤2: 移动到安全位置
        print(f"\n[2/5] 移动到安全位置: {self.safe_pose[:3]}")
        self._interpolate_move(self.safe_pose, timeout=3.0)
        time.sleep(0.5)

        # 步骤4: 等待用户放置peg
        print("\n" + "-" * 40)
        print(">>> 请将 peg 放入夹爪中 <<<")
        print(">>> 3秒后自动闭合夹爪 <<<")
        print("-" * 40)
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)

        # 步骤5: 闭合夹爪
        print("\n[5/5] 闭合夹爪...")
        self._close_gripper()
        time.sleep(0.8)

        print("\n" + "=" * 50)
        print("准备完成! 现在可以运行训练脚本")
        print("=" * 50)
        print("\n运行命令:")
        print("  终端1: ./run_learner.sh --debug")
        print("  终端2: ./run_actor.sh")
        print()


if __name__ == "__main__":
    preparer = PegPreparer()
    preparer.run()

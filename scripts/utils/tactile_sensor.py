"""
触觉传感器集成模块 - PaXini PX-6AX GEN3 MC-M2020-Elite
用于在数据采集时同步读取触觉 6 维力/力矩信息

基于官方通信协议文档 V1.0.5
- UART: 921600 baud, 8N1
- 功能码: 0x7B (读取应用区), 0x79 (写入配置区)
- 地址: 1008-1010 (合力), 1038+ (分布力阵列)
"""

import serial
import struct
import time
import numpy as np
from typing import Optional, Tuple
import glob


# 触觉传感器配置 - MC-M2020-Elite
PORT = '/dev/ttyACM0'
BAUD_RATE = 921600
DEV_ADDR = 0x01  # 设备地址 (模块号+1)

# 寄存器地址 (官方文档 Section 5.6.2)
ADDR_FORCE_SUM = 1008      # 合力 Fx, Fy, Fz (3字节)
ADDR_FORCE_ARRAY = 1038    # 分布力阵列起始地址

# MC-M2020-Elite 规格
TAXEL_COUNT = 9            # 测点数量
SCALE_FACTOR = 0.1         # 标定系数: 1 LSB = 0.1N

# 测点位置 (3x3 网格，单位: mm)
SENSOR_SPACING = 10.0
TAXEL_POSITIONS = [
    (-SENSOR_SPACING, -SENSOR_SPACING, 0),  # Taxel 1
    (0, -SENSOR_SPACING, 0),                # Taxel 2
    (SENSOR_SPACING, -SENSOR_SPACING, 0),   # Taxel 3
    (-SENSOR_SPACING, 0, 0),                # Taxel 4
    (0, 0, 0),                              # Taxel 5 (中心)
    (SENSOR_SPACING, 0, 0),                 # Taxel 6
    (-SENSOR_SPACING, SENSOR_SPACING, 0),   # Taxel 7
    (0, SENSOR_SPACING, 0),                 # Taxel 8
    (SENSOR_SPACING, SENSOR_SPACING, 0),    # Taxel 9
]


def calculate_lrc(data: bytes) -> int:
    """
    LRC校验位计算 (官方文档)
    LRC = (~sum + 1) & 0xFF = (0x100 - sum) & 0xFF
    """
    return ((0x100 - (sum(data) & 0xFF)) & 0xFF)


def find_sensor_port() -> Optional[str]:
    """自动查找传感器串口"""
    patterns = ['/dev/ttyACM*', '/dev/ttyUSB*']
    for pattern in patterns:
        ports = glob.glob(pattern)
        if ports:
            return sorted(ports)[0]
    return None


class TactileSensor:
    """
    PaXini PX-6AX GEN3 MC-M2020-Elite 触觉传感器

    通信协议:
    - 读取: 功能码 0xFB (0x7B | 0x80)
    - 写入: 功能码 0x79
    - 帧格式: 55 AA [len:2] [dev:1] [rsv:1] [func:1] [addr:4] [datalen:2] [data:N] [LRC:1]
    """

    def __init__(self, port: str = PORT, baud_rate: int = BAUD_RATE,
                 scale_factor: float = 0.1):
        """
        初始化触觉传感器

        Args:
            port: 串口设备路径
            baud_rate: 波特率
            scale_factor: 力值标定系数
        """
        self.port = port
        self.baud_rate = baud_rate
        self.scale_factor = scale_factor
        self.ser = None
        self._is_connected = False

    def connect(self) -> bool:
        """
        连接传感器

        Returns:
            是否成功连接
        """
        try:
            # 如果端口为 None，尝试自动查找
            port = self.port or find_sensor_port()
            if not port:
                print("✗ 未找到传感器串口设备")
                return False

            self.ser = serial.Serial(port, self.baud_rate, timeout=0.1)
            time.sleep(0.1)  # 等待串口稳定
            self._is_connected = True
            self.port = port
            print(f"✓ 触觉传感器已连接: {port}")
            return True
        except Exception as e:
            print(f"✗ 触觉传感器连接失败: {e}")
            self._is_connected = False
            return False

    def disconnect(self):
        """断开传感器"""
        if self.ser:
            self.ser.close()
            self._is_connected = False
            print("触觉传感器已断开")

    def is_connected(self) -> bool:
        """检查是否连接"""
        return self._is_connected

    def calibrate(self) -> bool:
        """
        校准传感器 (零点校准)

        在传感器无负载时调用此方法进行零点校准。
        写入地址3，值为1，功能码0x79

        Returns:
            是否成功
        """
        if not self._is_connected:
            print("✗ 传感器未连接")
            return False

        try:
            # 构建写入帧 (官方文档 Section 5.4.2 Write Request)
            # 功能码 0x79 (写入配置区)
            body = bytearray([DEV_ADDR, 0x00, 0x79])  # 设备地址, 保留, 功能码
            body += struct.pack('<I', 3)              # 地址 = 3 (校准寄存器)
            body += struct.pack('<H', 1)              # 数据长度 = 1
            body += bytes([0x01])                     # 数据 = 1 (触发校准)

            header = b'\x55\xAA'
            length = struct.pack('<H', len(body))
            frame = header + length + body
            frame += bytes([calculate_lrc(frame)])

            # Clear both input and output buffers before calibration
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            time.sleep(0.05)  # Wait for buffers to clear

            self.ser.write(frame)
            self.ser.flush()  # Ensure data is sent
            time.sleep(0.3)  # Wait longer for calibration to complete

            # Temporarily increase timeout for reading calibration response
            old_timeout = self.ser.timeout
            self.ser.timeout = 0.5
            response = self.ser.read(50)
            self.ser.timeout = old_timeout

            if len(response) >= 15 and response[:2] == b'\xAA\x55':
                status = response[13]
                if status == 0x00:
                    print("✓ 传感器校准成功")
                    return True
                else:
                    print(f"✗ 校准失败, 状态码: 0x{status:02X}")
                    return False
            else:
                # Debug: print what we actually received
                if len(response) > 0:
                    print(f"✗ 校准响应无效 (len={len(response)}, data={response[:20].hex()})")
                else:
                    print("✗ 校准无响应")
                return False

        except Exception as e:
            print(f"✗ 校准异常: {e}")
            return False

    def read_combined_force(self) -> Optional[np.ndarray]:
        """
        读取传感器内部计算的合力 (地址 1008-1010)

        Returns:
            [Fx, Fy, Fz] 单位 N，失败返回 None
        """
        if not self._is_connected:
            return None

        try:
            response = self._read_register(ADDR_FORCE_SUM, 3)
            if response is None:
                return None

            data = response[14:17]
            fx = struct.unpack('b', bytes([data[0]]))[0] * self.scale_factor
            fy = struct.unpack('b', bytes([data[1]]))[0] * self.scale_factor
            fz = struct.unpack('B', bytes([data[2]]))[0] * self.scale_factor

            return np.array([fx, fy, fz])

        except Exception:
            return None

    def read_force_torque(self) -> Optional[np.ndarray]:
        """
        读取一次 6 维力/力矩数据

        Returns:
            6 维数组 [Fx, Fy, Fz, Mx, My, Mz]，如果失败返回 None
        """
        if not self._is_connected:
            return None

        try:
            # 读取分布力数据
            response = self._read_register(ADDR_FORCE_ARRAY, TAXEL_COUNT * 3)
            if response is None:
                return np.zeros(6)

            raw_forces = self._parse_raw_forces(response)

            # 标定
            calibrated_forces = self._calibrate_forces(raw_forces)

            # 计算 6 维力/力矩
            force, torque = self._compute_force_torque(calibrated_forces)

            # 返回 6 维数组 [Fx, Fy, Fz, Mx, My, Mz]
            return np.concatenate([force, torque])

        except Exception as e:
            # 静默失败，返回零向量（避免打断数据采集）
            return np.zeros(6)

    def read_raw_taxels(self) -> Optional[list]:
        """
        读取原始测点数据

        Returns:
            [(Fx, Fy, Fz), ...] 9个测点的原始力值 (已缩放到N)
        """
        if not self._is_connected:
            return None

        try:
            response = self._read_register(ADDR_FORCE_ARRAY, TAXEL_COUNT * 3)
            if response is None:
                return None

            raw_forces = self._parse_raw_forces(response)
            return self._calibrate_forces(raw_forces)

        except Exception:
            return None

    def _read_register(self, addr: int, length: int) -> Optional[bytes]:
        """
        读取寄存器 (通用方法)

        Args:
            addr: 起始地址
            length: 读取字节数

        Returns:
            原始响应数据，失败返回 None
        """
        # 构建读取请求帧 (官方文档 Section 5.4.2)
        body = bytearray([DEV_ADDR, 0x00, 0xFB])   # 设备地址, 保留, 功能码(0x7B|0x80)
        body += struct.pack('<I', addr)            # 起始地址 (4字节, 小端)
        body += struct.pack('<H', length)          # 数据长度 (2字节, 小端)

        header = b'\x55\xAA'
        frame_len = struct.pack('<H', len(body))
        frame = header + frame_len + body
        frame += bytes([calculate_lrc(frame)])

        # 清空缓存并发送
        self.ser.reset_input_buffer()
        self.ser.write(frame)
        time.sleep(0.02)  # 等待传感器响应

        # 计算期望的响应长度
        # 响应格式: AA 55 [len:2] [dev:1] [rsv:1] [func:1] [addr:4] [datalen:2] [status:1] [data:N] [LRC:1]
        expected_len = 2 + 2 + 1 + 1 + 1 + 4 + 2 + 1 + length + 1  # = 15 + length
        response = self.ser.read(expected_len + 10)  # 多读一些以确保完整

        # 验证响应
        if len(response) < 15 + length:
            return None
        if response[:2] != b'\xAA\x55':
            return None

        return response

    def _parse_raw_forces(self, response: bytes):
        """解析原始力数据"""
        # 数据从字节 14 开始 (status 之后)
        raw_payload = response[14:14 + TAXEL_COUNT * 3]
        forces = []

        for i in range(TAXEL_COUNT):
            # Fx, Fy: 有符号 (-128~+127), Fz: 无符号 (0~255)
            fx_raw = struct.unpack('b', bytes([raw_payload[i*3]]))[0]
            fy_raw = struct.unpack('b', bytes([raw_payload[i*3 + 1]]))[0]
            fz_raw = struct.unpack('B', bytes([raw_payload[i*3 + 2]]))[0]
            forces.append((fx_raw, fy_raw, fz_raw))

        return forces

    def _calibrate_forces(self, raw_forces):
        """将原始力值转换为物理单位 (N)"""
        calibrated = []
        for fx_raw, fy_raw, fz_raw in raw_forces:
            fx = fx_raw * self.scale_factor
            fy = fy_raw * self.scale_factor
            fz = fz_raw * self.scale_factor
            calibrated.append((fx, fy, fz))

        return calibrated

    def _compute_force_torque(self, forces) -> Tuple[np.ndarray, np.ndarray]:
        """从分布力计算 6 维力/力矩"""
        force = np.zeros(3)
        torque = np.zeros(3)

        for (fx, fy, fz), (x, y, z) in zip(forces, TAXEL_POSITIONS):
            f_vec = np.array([fx, fy, fz])
            r_vec = np.array([x, y, z])

            # 合力: F_total = Σ F_i
            force += f_vec

            # 力矩: τ = Σ (r_i × F_i)
            torque += np.cross(r_vec, f_vec)

        return force, torque


# 便捷函数：用于快速创建和连接传感器
def create_tactile_sensor(port: str = PORT, scale_factor: float = 0.1,
                          auto_connect: bool = True) -> Optional[TactileSensor]:
    """
    创建并连接触觉传感器

    Args:
        port: 串口设备路径
        scale_factor: 力值标定系数
        auto_connect: 是否自动连接

    Returns:
        TactileSensor 实例，如果连接失败返回 None
    """
    sensor = TactileSensor(port=port, scale_factor=scale_factor)

    if auto_connect:
        if sensor.connect():
            return sensor
        else:
            return None

    return sensor

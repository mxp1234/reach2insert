#!/usr/bin/env python3
"""
触觉传感器实时测试与可视化脚本

用于调试 PaXini PX-6AX GEN3 MC-M2020-Elite 触觉传感器

功能:
1. 实时显示 6D 力/力矩数据
2. 实时绘图可视化
3. 原始测点数据查看
4. 校准功能

使用方法:
=========

1. 基本测试 (终端实时输出):
   python test_tactile_sensor.py

2. 实时绘图可视化:
   python test_tactile_sensor.py --plot

3. 查看原始测点数据:
   python test_tactile_sensor.py --raw

4. 校准传感器:
   python test_tactile_sensor.py --calibrate

5. 指定串口:
   python test_tactile_sensor.py --port /dev/ttyACM1

6. 调试模式 (显示原始字节):
   python test_tactile_sensor.py --debug
"""

import sys
import time
import argparse
import numpy as np

# 添加传感器模块路径
sys.path.insert(0, "/home/pi-zero/Documents/openpi/third_party/real_franka")

try:
    from data_collection.tactile_sensor import TactileSensor, find_sensor_port
    SENSOR_AVAILABLE = True
except ImportError as e:
    print(f"Error importing TactileSensor: {e}")
    SENSOR_AVAILABLE = False


def test_basic(sensor: TactileSensor, duration: float = 30.0, rate: float = 20.0):
    """
    基本测试 - 终端实时输出
    """
    print("\n" + "=" * 70)
    print("  Tactile Sensor Basic Test")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")

    dt = 1.0 / rate
    start_time = time.time()
    read_count = 0
    error_count = 0

    try:
        while time.time() - start_time < duration:
            t_loop = time.time()

            # 读取 6D 力/力矩
            data = sensor.read_force_torque()

            if data is not None:
                fx, fy, fz, mx, my, mz = data
                f_mag = np.sqrt(fx**2 + fy**2 + fz**2)
                m_mag = np.sqrt(mx**2 + my**2 + mz**2)

                # 检查是否全为零
                is_zero = np.allclose(data, 0, atol=1e-6)
                status = "⚠ ALL ZERO" if is_zero else "OK"

                print(f"\r[{status}] F: [{fx:+7.3f}, {fy:+7.3f}, {fz:+7.3f}] N  |  "
                      f"τ: [{mx:+7.2f}, {my:+7.2f}, {mz:+7.2f}] N·mm  |  "
                      f"|F|={f_mag:.3f} N  |τ|={m_mag:.2f} N·mm", end='')
                read_count += 1
            else:
                print(f"\r[ERROR] Failed to read sensor data (errors: {error_count})", end='')
                error_count += 1

            # 控制读取频率
            elapsed = time.time() - t_loop
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass

    print(f"\n\nTest finished:")
    print(f"  Successful reads: {read_count}")
    print(f"  Errors: {error_count}")
    print(f"  Duration: {time.time() - start_time:.1f}s")


def test_raw_taxels(sensor: TactileSensor, duration: float = 30.0, rate: float = 10.0):
    """
    原始测点数据测试 - 显示 9 个测点的独立读数
    """
    print("\n" + "=" * 70)
    print("  Tactile Sensor Raw Taxel Test (9 taxels)")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")

    dt = 1.0 / rate

    try:
        while True:
            t_loop = time.time()

            taxels = sensor.read_raw_taxels()

            if taxels is not None:
                print("\033[2J\033[H")  # 清屏
                print("=" * 70)
                print("  Raw Taxel Data (9 taxels, 3x3 grid)")
                print("=" * 70)
                print("\n  Taxel Layout (top view):")
                print("  ┌─────────┬─────────┬─────────┐")

                for row in range(3):
                    row_data = []
                    for col in range(3):
                        idx = row * 3 + col
                        fx, fy, fz = taxels[idx]
                        row_data.append(f"Fz={fz:+.2f}")
                    print(f"  │ {row_data[0]:^7} │ {row_data[1]:^7} │ {row_data[2]:^7} │")
                    if row < 2:
                        print("  ├─────────┼─────────┼─────────┤")

                print("  └─────────┴─────────┴─────────┘")

                print("\n  Detailed taxel readings:")
                print("  " + "-" * 50)
                for i, (fx, fy, fz) in enumerate(taxels):
                    is_zero = abs(fx) < 0.01 and abs(fy) < 0.01 and abs(fz) < 0.01
                    status = "⚠" if is_zero else "✓"
                    print(f"  {status} Taxel {i+1}: Fx={fx:+6.3f}, Fy={fy:+6.3f}, Fz={fz:+6.3f} N")

                # 统计
                all_zero = all(abs(fx) < 0.01 and abs(fy) < 0.01 and abs(fz) < 0.01
                              for fx, fy, fz in taxels)
                if all_zero:
                    print("\n  ⚠ WARNING: All taxels reading zero!")
                    print("    Possible causes:")
                    print("    - Sensor not pressed")
                    print("    - Needs calibration (run with --calibrate)")
                    print("    - Communication issue")

            else:
                print("\r[ERROR] Failed to read taxel data", end='')

            elapsed = time.time() - t_loop
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass

    print("\nTest finished.")


def test_plot(sensor: TactileSensor, duration: float = 60.0, rate: float = 30.0):
    """
    实时绘图可视化
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Error: matplotlib not installed. Run: pip install matplotlib")
        return

    print("\n" + "=" * 70)
    print("  Tactile Sensor Real-time Plot")
    print("=" * 70)
    print("Close the plot window to stop\n")

    # 数据缓冲区
    history_len = 200
    time_data = np.zeros(history_len)
    force_data = np.zeros((history_len, 3))  # Fx, Fy, Fz
    torque_data = np.zeros((history_len, 3))  # Mx, My, Mz

    start_time = time.time()
    data_idx = [0]  # 使用列表以便在闭包中修改

    # 创建图形
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Tactile Sensor Real-time Data', fontsize=14)

    # 力曲线
    line_fx, = axes[0].plot([], [], 'r-', label='Fx', linewidth=1.5)
    line_fy, = axes[0].plot([], [], 'g-', label='Fy', linewidth=1.5)
    line_fz, = axes[0].plot([], [], 'b-', label='Fz', linewidth=1.5)
    axes[0].set_ylabel('Force (N)')
    axes[0].set_xlim(0, history_len / rate)
    axes[0].set_ylim(-5, 10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Force')

    # 力矩曲线
    line_mx, = axes[1].plot([], [], 'r-', label='Mx', linewidth=1.5)
    line_my, = axes[1].plot([], [], 'g-', label='My', linewidth=1.5)
    line_mz, = axes[1].plot([], [], 'b-', label='Mz', linewidth=1.5)
    axes[1].set_ylabel('Torque (N·mm)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_xlim(0, history_len / rate)
    axes[1].set_ylim(-50, 50)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Torque')

    # 状态文本
    status_text = axes[0].text(0.02, 0.95, '', transform=axes[0].transAxes,
                               fontsize=10, verticalalignment='top',
                               fontfamily='monospace')

    def init():
        line_fx.set_data([], [])
        line_fy.set_data([], [])
        line_fz.set_data([], [])
        line_mx.set_data([], [])
        line_my.set_data([], [])
        line_mz.set_data([], [])
        return line_fx, line_fy, line_fz, line_mx, line_my, line_mz, status_text

    def update(frame):
        nonlocal time_data, force_data, torque_data

        # 读取传感器数据
        data = sensor.read_force_torque()

        if data is not None:
            # 滚动数据
            time_data = np.roll(time_data, -1)
            force_data = np.roll(force_data, -1, axis=0)
            torque_data = np.roll(torque_data, -1, axis=0)

            # 更新最新值
            current_time = time.time() - start_time
            time_data[-1] = current_time
            force_data[-1] = data[:3]
            torque_data[-1] = data[3:]

            # 计算有效数据范围
            valid_start = max(0, data_idx[0] - history_len + 1)
            x_data = np.arange(history_len) / rate

            # 更新曲线
            line_fx.set_data(x_data, force_data[:, 0])
            line_fy.set_data(x_data, force_data[:, 1])
            line_fz.set_data(x_data, force_data[:, 2])
            line_mx.set_data(x_data, torque_data[:, 0])
            line_my.set_data(x_data, torque_data[:, 1])
            line_mz.set_data(x_data, torque_data[:, 2])

            # 自动调整 Y 轴范围
            f_min = np.min(force_data) - 0.5
            f_max = np.max(force_data) + 0.5
            axes[0].set_ylim(min(f_min, -1), max(f_max, 1))

            t_min = np.min(torque_data) - 5
            t_max = np.max(torque_data) + 5
            axes[1].set_ylim(min(t_min, -10), max(t_max, 10))

            # 状态文本
            is_zero = np.allclose(data, 0, atol=0.01)
            status = "⚠ ALL ZERO - Check sensor!" if is_zero else "OK"
            status_text.set_text(f"Status: {status}\n"
                                f"F: [{data[0]:+.2f}, {data[1]:+.2f}, {data[2]:+.2f}] N\n"
                                f"τ: [{data[3]:+.1f}, {data[4]:+.1f}, {data[5]:+.1f}] N·mm")
            if is_zero:
                status_text.set_color('red')
            else:
                status_text.set_color('green')

            data_idx[0] += 1

        return line_fx, line_fy, line_fz, line_mx, line_my, line_mz, status_text

    # 创建动画
    ani = FuncAnimation(fig, update, init_func=init,
                       interval=int(1000/rate), blit=True, cache_frame_data=False)

    plt.tight_layout()
    plt.show()


def test_debug(sensor: TactileSensor):
    """
    调试模式 - 显示原始通信数据
    """
    import struct

    print("\n" + "=" * 70)
    print("  Tactile Sensor Debug Mode")
    print("=" * 70)
    print("Press Ctrl+C to stop\n")

    # 直接访问串口进行原始读取
    ser = sensor.ser

    # 构建读取请求
    DEV_ADDR = 0x01
    ADDR_FORCE_ARRAY = 1038
    TAXEL_COUNT = 9

    def calculate_lrc(data):
        return ((0x100 - (sum(data) & 0xFF)) & 0xFF)

    body = bytearray([DEV_ADDR, 0x00, 0xFB])
    body += struct.pack('<I', ADDR_FORCE_ARRAY)
    body += struct.pack('<H', TAXEL_COUNT * 3)

    header = b'\x55\xAA'
    frame_len = struct.pack('<H', len(body))
    frame = header + frame_len + body
    frame += bytes([calculate_lrc(frame)])

    print(f"Request frame ({len(frame)} bytes):")
    print(f"  Hex: {frame.hex(' ')}")
    print(f"  Breakdown:")
    print(f"    Header: {frame[:2].hex()}")
    print(f"    Length: {struct.unpack('<H', frame[2:4])[0]}")
    print(f"    DevAddr: {frame[4]:02x}")
    print(f"    Reserved: {frame[5]:02x}")
    print(f"    FuncCode: {frame[6]:02x}")
    print(f"    Address: {struct.unpack('<I', frame[7:11])[0]}")
    print(f"    DataLen: {struct.unpack('<H', frame[11:13])[0]}")
    print(f"    LRC: {frame[13]:02x}")
    print()

    try:
        count = 0
        while count < 10:
            ser.reset_input_buffer()
            ser.write(frame)
            time.sleep(0.05)

            response = ser.read(100)

            print(f"[{count+1}] Response ({len(response)} bytes):")
            if len(response) > 0:
                print(f"  Hex: {response.hex(' ')}")

                if len(response) >= 2:
                    print(f"  Header: {response[:2].hex()} (expected: aa 55)")

                if len(response) >= 4:
                    resp_len = struct.unpack('<H', response[2:4])[0]
                    print(f"  Body length: {resp_len}")

                if len(response) >= 14:
                    print(f"  DevAddr: {response[4]:02x}")
                    print(f"  FuncCode: {response[6]:02x}")
                    addr = struct.unpack('<I', response[7:11])[0]
                    print(f"  Address: {addr}")
                    data_len = struct.unpack('<H', response[11:13])[0]
                    print(f"  DataLen: {data_len}")
                    status = response[13]
                    print(f"  Status: {status:02x} ({'OK' if status == 0 else 'ERROR'})")

                if len(response) >= 14 + TAXEL_COUNT * 3:
                    print(f"  Raw data bytes: {response[14:14+TAXEL_COUNT*3].hex(' ')}")

                    # 解析测点数据
                    print(f"  Parsed taxels:")
                    for i in range(TAXEL_COUNT):
                        fx = struct.unpack('b', bytes([response[14 + i*3]]))[0]
                        fy = struct.unpack('b', bytes([response[14 + i*3 + 1]]))[0]
                        fz = struct.unpack('B', bytes([response[14 + i*3 + 2]]))[0]
                        print(f"    Taxel {i+1}: raw=({fx:+4d}, {fy:+4d}, {fz:3d}) -> "
                              f"({fx*0.1:+.1f}, {fy*0.1:+.1f}, {fz*0.1:.1f}) N")
            else:
                print("  (empty)")

            print()
            count += 1
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass

    print("Debug finished.")


def calibrate_sensor(sensor: TactileSensor):
    """
    校准传感器
    """
    print("\n" + "=" * 70)
    print("  Tactile Sensor Calibration")
    print("=" * 70)
    print("\nIMPORTANT: Make sure the sensor is NOT under load!")
    print("The sensor surface should be free and not touching anything.\n")

    input("Press Enter to calibrate...")

    print("\nCalibrating...")
    success = sensor.calibrate()

    if success:
        print("\n✓ Calibration successful!")
        print("\nVerifying with a test read...")
        time.sleep(0.5)

        data = sensor.read_force_torque()
        if data is not None:
            print(f"  Current reading: F=[{data[0]:.3f}, {data[1]:.3f}, {data[2]:.3f}] N")
            print(f"                   τ=[{data[3]:.2f}, {data[4]:.2f}, {data[5]:.2f}] N·mm")

            if np.allclose(data, 0, atol=0.5):
                print("  ✓ Reading is near zero - calibration looks good!")
            else:
                print("  ⚠ Reading is not zero - this may be normal if there's residual contact")
    else:
        print("\n✗ Calibration failed!")
        print("  Please check:")
        print("  - Sensor connection")
        print("  - Serial port permissions")
        print("  - Sensor power")


def main():
    parser = argparse.ArgumentParser(description='Tactile Sensor Test Tool')
    parser.add_argument('--port', type=str, default='/dev/ttyACM1',
                        help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--plot', action='store_true',
                        help='Show real-time plot')
    parser.add_argument('--raw', action='store_true',
                        help='Show raw taxel data')
    parser.add_argument('--calibrate', action='store_true',
                        help='Calibrate sensor (zero offset)')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode (show raw bytes)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Test duration in seconds (default: 60)')
    parser.add_argument('--rate', type=float, default=20.0,
                        help='Sampling rate in Hz (default: 20)')
    parser.add_argument('--auto-port', action='store_true',
                        help='Auto-detect sensor port')

    args = parser.parse_args()

    if not SENSOR_AVAILABLE:
        print("Error: TactileSensor module not available!")
        print("Make sure the path is correct:")
        print("  /home/pi-zero/Documents/openpi/third_party/real_franka/data_collection/tactile_sensor.py")
        return

    # 确定串口
    port = args.port
    if args.auto_port:
        detected_port = find_sensor_port()
        if detected_port:
            port = detected_port
            print(f"Auto-detected port: {port}")
        else:
            print("Warning: No sensor port auto-detected, using default")

    print(f"\nConnecting to tactile sensor on {port}...")

    # 创建并连接传感器
    sensor = TactileSensor(port=port)
    if not sensor.connect():
        print("\nFailed to connect!")
        print("\nTroubleshooting:")
        print("  1. Check if device exists: ls -la /dev/ttyACM*")
        print("  2. Check permissions: sudo chmod 666 /dev/ttyACM0")
        print("  3. Add user to dialout group: sudo usermod -a -G dialout $USER")
        print("  4. Try a different port: --port /dev/ttyACM1")
        return

    try:
        if args.calibrate:
            calibrate_sensor(sensor)
        elif args.debug:
            test_debug(sensor)
        elif args.plot:
            test_plot(sensor, duration=args.duration, rate=args.rate)
        elif args.raw:
            test_raw_taxels(sensor, duration=args.duration, rate=args.rate)
        else:
            test_basic(sensor, duration=args.duration, rate=args.rate)
    finally:
        sensor.disconnect()


if __name__ == "__main__":
    main()

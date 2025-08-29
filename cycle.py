import os
import time
import json
import threading
import pygame
import pyaudio
import torch
import webrtcvad
import numpy as np
import re
from queue import Queue, Empty
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info 
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from scipy.signal import resample_poly 
from collections import deque
import paho.mqtt.client as mqtt

# 清空CUDA缓存（仅保留必要一次）
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# 配置参数（仅保留核心必要参数）
AUDIO_CHANNELS = 1
AUDIO_RATE = 48000 
AUDIO_CHUNK = 1024
VAD_MODE = 3
NO_SPEECH_THRESHOLD = 4
TTS_RATE = 150
TTS_VOLUME = 0.9
POINT_JSON_DIR = "point_json"
os.makedirs(POINT_JSON_DIR, exist_ok=True)
TASK_POINTS_FILE = os.path.join(POINT_JSON_DIR, "task_points.json")

# 命令类型定义（仅保留核心导航相关类型）
class CommandType:
    NAVIGATE = 9  # 合并导航类型（含循环控制）


class VoiceNavigationSystem:
    def __init__(self):
        # 核心参数配置
        self.mqtt_broker = '10.42.0.1'
        self.mqtt_port = 1883
        self.mqtt_control_topic = 'robot_control'
        self.mqtt_status_topic = 'base_status'
        self.qwen_model_path = '/home/y/LLM/Qwen2-VL-2B-Instruct'
        self.sensevoice_model_path = '/home/y/LLM/SenseVoiceSmall'
        print(f"📌 任务点文件路径: {TASK_POINTS_FILE}")

        # 设备初始化
        self.device = self._initialize_device()
        self.audio_stream = None
        self.pyaudio_instance = None

        # 核心组件与锁
        self.mqtt_client = self.initialize_mqtt()
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.audio_block_lock = threading.Lock()
        self.is_running = True  # 系统运行标志
        self.FORMAT = pyaudio.paInt16

        # 循环导航状态（核心）
        self.is_cycling = False
        self.current_cycle_step = 0
        self.cycle_order = [1, 2, 3, 0]  # 1→2→3→0循环顺序
        self.completed_cycles = 0
        self.max_cycles = 1

        # 任务点与位姿
        self.task_points = self._load_task_points()
        self.current_pose = None

        # 初始化依赖
        pygame.mixer.init()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)
        self.initialize_tts()  # 初始化TTS
        print("🤖 智能语音导航系统初始化完成")

    def _initialize_device(self):
        """设备初始化（简化逻辑）"""
        try:
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
                torch.tensor([1.0], device=dev)  # 验证设备
                return dev
            else:
                print("CUDA不可用，使用CPU")
                return torch.device("cpu")
        except Exception as e:
            print(f"设备初始化失败: {e}，强制使用CPU")
            return torch.device("cpu")

    def initialize_mqtt(self):
        """MQTT初始化（直接返回客户端对象）"""
        try:
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
            client.on_connect = self.on_connect
            client.on_message = self.on_message
            client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            client.loop_start()
            print("✅ MQTT连接成功")
            return client
        except Exception as e:
            print(f"❌ MQTT连接失败: {e}")
            raise Exception("MQTT初始化失败，系统无法运行")

    def initialize_tts(self):
        """TTS初始化（保留核心功能）"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init(driverName='espeak')
            self.tts_engine.setProperty('rate', TTS_RATE)
            self.tts_engine.setProperty('volume', TTS_VOLUME)

            # 选择中文语音
            for voice in self.tts_engine.getProperty('voices'):
                for lang in voice.languages:
                    if isinstance(lang, bytes) and lang.decode('utf-8').startswith(('cmn', 'zh')):
                        self.tts_engine.setProperty('voice', voice.id)
                        print(f"✅ 成功设置中文语音：{voice.id}")
                        return
            print("⚠️ 未找到中文语音，可能导致播报异常")
        except Exception as e:
            print(f"❌ TTS初始化失败: {e}")
            self.tts_engine = None

    def speak(self, text):
        """语音播报（简化过滤逻辑）"""
        filtered_text = re.sub(r'[^\u4e00-\u9fa5，。？！,.;?!\s]', '', text).strip()
        if not filtered_text or not self.tts_engine:
            print(f"⚠️ 无法播报：{text}（无有效文本或TTS未初始化）")
            return

        def _speak_offline():
            try:
                with self.tts_lock, self.audio_block_lock:
                    self.tts_engine.say(filtered_text)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ 播报错误: {e}")
        threading.Thread(target=_speak_offline, daemon=True).start()

    # ------------------------------
    # 任务点管理（核心）
    # ------------------------------
    def _load_task_points(self):
        """从文件加载任务点（简化逻辑）"""
        if os.path.exists(TASK_POINTS_FILE):
            try:
                with open(TASK_POINTS_FILE, 'r', encoding='utf-8') as f:
                    points = json.load(f)
                    print(f"✅ 加载任务点：{list(points.keys())}")
                    return points
            except Exception as e:
                print(f"❌ 加载任务点失败: {e}")
        print("⚠️ 未找到任务点文件，初始化为空")
        return {}

    # ------------------------------
    # 导航与控制（核心）
    # ------------------------------
    def send_goal_point(self, point_id):
        """发送目标点（基于绝对位移）"""
        point_key = str(point_id)
        if point_key not in self.task_points:
            self.speak(f"未找到{point_id}号任务点")
            return False

        point_info = self.task_points[point_key]
        message = {
            "cmd_type": "goal_point_set",
            "x": point_info["pose"]["x"],
            "y": point_info["pose"]["y"],
            "z": point_info["pose"]["yaw"]
        }

        try:
            cmd_json = json.dumps(message)
            # 连续发送3次确保送达
            for _ in range(3):
                self.mqtt_client.publish(self.mqtt_control_topic, cmd_json, qos=2)
                time.sleep(0.1)

            # 循环模式语音提示
            if self.is_cycling:
                step = self.current_cycle_step + 1
                self.speak(f"已前往{point_id}号点（循环步骤{step}/4），等待机械臂完成工作")
            else:
                self.speak(f"已发送前往{point_id}号点的指令")

            # 发送机械臂动作指令
            self.send_lerobot(point_id)
            print(f"✅ 发送目标点{point_id}: {message}")
            return True
        except Exception as e:
            print(f"❌ 发送目标点失败: {e}")
            self.speak("发送指令失败，请重试")
            return False

    def start_cycle_navigation(self):
        """启动单次循环任务（1→2→3→0）"""
        # 校验任务点完整性
        required_points = [str(p) for p in self.cycle_order]
        missing = [p for p in required_points if p not in self.task_points]
        if missing:
            self.speak(f"循环启动失败：缺少任务点{','.join(missing)}")
            self.is_cycling = False
            return

        # 重置循环状态
        self.is_cycling = True
        self.current_cycle_step = 0
        self.completed_cycles = 0
        self.speak("启动单次循环任务（1→2→3→0），完成后自动停止")

        # 发送循环启动信号（机械臂）
        for _ in range(3):
            self.send_lerobot(6)
            time.sleep(0.2)

        # 发送第一个目标点（1号点）
        first_point = str(self.cycle_order[0])
        self.send_goal_point(first_point)

    def send_lerobot(self, number):
        """向机械臂发送指令（简化逻辑）"""
        try:
            if not isinstance(number, int) or number < 0:
                raise ValueError(f"无效数字: {number}")

            message = {"type": "greeting", "number": number}
            cmd_json = json.dumps(message)
            # 连续发送3次确保送达
            for _ in range(3):
                self.mqtt_client.publish("lerobot_status", cmd_json, qos=2)
                time.sleep(0.1)
            print(f"✅ 向lerobot_status发送数字: {number}")

            # 打招呼专属播报
            if number == 9:
                self.speak("你好，我是小车，需要帮助吗")
        except Exception as e:
            print(f"❌ 发送机械臂指令失败: {e}")
            self.speak("操作失败，请重试")

    # ------------------------------
    # MQTT回调（核心）
    # ------------------------------
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"MQTT连接成功,状态码: {rc}")
        # 订阅必要话题
        client.subscribe([
            (self.mqtt_status_topic, 0),
            (self.mqtt_control_topic, 0),
            ("script/grab_status", 0)
        ])

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())

            # 处理机械臂状态（循环导航中）
            if msg.topic == "script/grab_status" and "grab_successfully" in payload and self.is_cycling:
                arm_status = payload.get('grab_successfully', -1)
                current_point = self.cycle_order[self.current_cycle_step]
                print(f"从 {msg.topic} 收到机械臂状态: {arm_status}（当前点：{current_point}，步骤：{self.current_cycle_step}）")

                # 状态→目标点映射（核心逻辑）
                status_to_target = { 1:2,
                                     2:3, 
                                     3:0, 
                                     0:None, 
                                     -1:None}
                target_point = status_to_target.get(arm_status, None)

                # 状态分支处理
                if arm_status == -1:
                    self.speak("未获取到机械臂有效状态，循环暂停")
                    self.is_cycling = False
                elif arm_status == 0:
                    self.completed_cycles += 1
                    self.speak(f"已完成单次循环（1→2→3→0），共{self.completed_cycles}次")
                    self.is_cycling = False
                elif target_point is not None:
                    try:
                        next_step = self.cycle_order.index(target_point)
                        self.speak(f"机械臂在{current_point}号点完成，前往{target_point}号点（步骤{next_step+1}/4）")
                        self.current_cycle_step = next_step
                        if str(target_point) not in self.task_points:
                            raise ValueError(f"目标点{target_point}不存在")
                        self.send_goal_point(target_point)
                    except (ValueError, Exception) as e:
                        self.speak(f"循环中断：{str(e)}")
                        self.is_cycling = False
                else:
                    self.speak(f"收到未知状态{arm_status}，循环暂停")
                    self.is_cycling = False
                return

            # 处理底盘位姿更新
            if msg.topic == self.mqtt_status_topic and "pose" in payload:
                pose = payload["pose"]
                self.current_pose = (pose["x"], pose["y"], pose["yaw"])
                # print(f"更新位姿: x={pose['x']:.2f}, y={pose['y']:.2f}, yaw={pose['yaw']:.2f}")

        except json.JSONDecodeError:
            print("❌ MQTT消息解析失败")
            self.speak("指令解析错误")
        except Exception as e:
            print(f"❌ 处理MQTT消息错误: {e}")

    # ------------------------------
    # 系统启停（核心）
    # ------------------------------
    def stop_system(self):
        """资源释放（简化逻辑）"""
        print("\n开始关闭系统...")
        self.is_running = False
        self.is_cycling = False

        # 停止音频播放
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

        # 关闭MQTT
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        # 关闭音频流
        if self.audio_stream and self.pyaudio_instance:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.pyaudio_instance.terminate()

        # 清理CUDA
        if self.device and self.device.type == 'cuda':
            torch.cuda.empty_cache()

        print("✅ 系统已完全关闭")

    def run(self):
        """系统运行入口"""
        try:
            self.start_cycle_navigation()  # 启动循环
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.speak("收到退出信号，再见")
            time.sleep(1)
        finally:
            self.stop_system()

def main():
    try:
        nav_system = VoiceNavigationSystem()
        nav_system.run()
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")

if __name__ == "__main__":
    main()
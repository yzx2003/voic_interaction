import os
import time
import json
import wave
import threading
import cv2
from PIL import Image 
import pygame
import pyaudio
import torch
import webrtcvad
import numpy as np
import re
from queue import Queue, Empty
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info 
from modelscope import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from transformers import Qwen2VLForConditionalGeneration
from transformers import BitsAndBytesConfig

import torch
torch.cuda.empty_cache()  # 清空缓存
torch.backends.cudnn.benchmark = True

from scipy.signal import resample_poly 
from collections import deque

# 配置参数
AUDIO_CHANNELS = 1
AUDIO_RATE = 48000 
AUDIO_CHUNK = 1024
VAD_MODE = 3 #1 #2 #3
NO_SPEECH_THRESHOLD = 4
TTS_RATE = 150
TTS_VOLUME = 0.9
POINT_JSON_DIR = "point_json"
os.makedirs(POINT_JSON_DIR, exist_ok=True)
TASK_POINTS_FILE = os.path.join(POINT_JSON_DIR, "task_points.json")  # 任务点保存文件

# 命令类型定义
class CommandType:
    A_TO_B = 1   # 常规三个任务点
    B_TO_C = 2
    C_TO_D = 3
    D_TO_A = 4   # 起始点
    IMAGE_RECOGNITION = 5 # 图像检测
    DIALOGUE = 6  # 对话类型
    ADD_TASK_POINT = 7  # 添加任务点
    MODIFY_TASK_POINT = 8  # 修改任务点
    NAVIGATE = 9  # 合并导航类型（包含按描述导航）
    GREET = 10  # 打个招呼


class VoiceNavigationSystem:
    def __init__(self):
        # 参数设置
        self.mqtt_broker = '10.42.0.1' #本地 localhost  10.42.0.1
        self.mqtt_port = 1883
        self.mqtt_control_topic = 'robot_control'  # 控制指令话题
        self.mqtt_status_topic = 'base_status'     # 底盘状态话题
        self.qwen_model_path = '/home/y/LLM/Qwen2-VL-2B-Instruct'
        self.sensevoice_model_path = '/home/y/LLM/SenseVoiceSmall'
        print(f"📌 任务点文件路径: {TASK_POINTS_FILE}")
        # 设备检测与初始化
        self.device = None
        self._initialize_device()

        self.audio_stream = None
        self.pyaudio_instance = None
        
        # 初始化组件
        self.mqtt_client = None
        self.initialize_mqtt()
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        self.command_queue = Queue()
        self.is_running = False
        self.is_listening = False
        self.FORMAT = pyaudio.paInt16
        self.tts_lock = threading.Lock()
        self.audio_block_lock = threading.Lock() 

        # 音频视频队列
        self.audio_queue = Queue()
        self.video_queue = Queue(maxsize=100) # 限制队列大小，避免阻塞
        self.video_buffer = deque(maxlen=300)  # 存储10秒视频(按30fps)
        self.last_active_time = time.time()
        self.recording_active = True
        self.segments_to_save = []
        self.saved_intervals = []
        self.last_vad_end_time = 0

        # 初始化VAD
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(VAD_MODE)
        
        # 任务点管理（ID: {x, y, z, description}）
        self.task_points = self._load_task_points()  # 从文件加载
        self.current_pose = None  # 存储当前底盘位姿 (x, y, yaw)
        self.last_nav_command = None  # 保存最后一个导航指令
        self.last_nav_query = ""  # 保存最后一个导航查询文本

        self.is_cycling = False  # 循环导航标志位
        self.current_cycle_step = 0  # 循环步骤计数器
        self.cycle_order = [1, 2, 3, 0]  # 循环顺序
        self.completed_cycles = 0  # 已完成的任务循环次数
        self.max_cycles = 1  # 最大任务循环次数（固定为1）

        pygame.mixer.init()
        
        # 命令映射（合并导航相关指令）
        self.command_mapping = {
           
            # 新增任务点指令
            "添加任务点": CommandType.ADD_TASK_POINT,
            "增加任务点": CommandType.ADD_TASK_POINT,
            "保存当前点": CommandType.ADD_TASK_POINT,
            "记录这个点": CommandType.ADD_TASK_POINT,
            
            # 修改任务点指令
            "设定为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "设置为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "置为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "设为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "改为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "修改为一号点": (CommandType.MODIFY_TASK_POINT, 1),
            "为一号点": (CommandType.MODIFY_TASK_POINT, 1),

            "设定为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "设置为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "设为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "置为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "改为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "修改为二号点": (CommandType.MODIFY_TASK_POINT, 2),
            "为二号点": (CommandType.MODIFY_TASK_POINT, 2),

            "设定为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "设置为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "置为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "设为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "改为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "修改为三号点": (CommandType.MODIFY_TASK_POINT, 3),
            "为三号点": (CommandType.MODIFY_TASK_POINT, 3),

            "设定为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "设置为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "置为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "设为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "改为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "修改为起始点": (CommandType.MODIFY_TASK_POINT, 0),
            "为起始点": (CommandType.MODIFY_TASK_POINT, 0),

            # 打招呼指令映射
            "打个招呼": (CommandType.GREET, 9),
            "打个": (CommandType.GREET, 9),
            "招呼": (CommandType.GREET, 9),
            "问候一下": (CommandType.GREET, 9),
            "问候": (CommandType.GREET, 9),
            "问好": (CommandType.GREET, 9),
        }
        
        self.welcome_text = "大家好，我是语音导航助手，请说出您的指令"
        self.listening_text = "我在听，请说话"

        print("🤖 智能语音导航系统初始化中...")
        self.initialize_system()

    def _initialize_device(self):
        try:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"CUDA可用，使用GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("CUDA不可用，使用CPU")
            test_tensor = torch.tensor([1.0], device=self.device)
            return True
        except Exception as e:
            print(f"设备初始化失败: {e}")
            self.device = torch.device("cpu")
            print("强制使用CPU作为 fallback")
            return False

    def initialize_models(self):
        try:
            if self.device is None:
                self._initialize_device()
                
            # 加载千问VL模型（用于图像识别和场景描述）
            print("加载千问VL模型...")
            min_pixels = 128 * 28 * 28
            max_pixels = 256 * 28 * 28

            # 8bit量化（平衡速度和精度）
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dype=torch.float16
            )
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.qwen_model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.qwen_model.eval()

            self.qwen_processor = AutoProcessor.from_pretrained(
                self.qwen_model_path,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                trust_remote_code=True
            )

            # 加载语音识别模型
            print("加载SenseVoice模型...")
            self.sensevoice_pipeline = pipeline(
                task=Tasks.auto_speech_recognition,
                model=self.sensevoice_model_path,
                model_revision="v1.0.0",
                trust_remote_code=True
            )
            print("SenseVoice模型加载成功")
                
            print(f"所有模型已成功加载到 {self.device}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            print(f"错误详情: {traceback.format_exc()}")
            return False

    def initialize_mqtt(self):
        try:
            import paho.mqtt.client as mqtt
            self.mqtt_client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
            self.mqtt_client.on_connect = self.on_connect
            self.mqtt_client.on_message = self.on_message
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f"❌ MQTT连接失败: {e}")
            return False

    def initialize_tts(self):
        """初始化离线TTS引擎"""
        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init(driverName='espeak')
            self.tts_engine.setProperty('rate', TTS_RATE)
            self.tts_engine.setProperty('volume', TTS_VOLUME)
            
            voices = self.tts_engine.getProperty('voices')
            print("可用语音列表：")
            for i, voice in enumerate(voices):
                print(f"索引 {i}：ID={voice.id}，名称={voice.name}，语言={voice.languages}")
            
            # 选择中文语音
            selected_voice = None
            if len(voices) > 12:  # 优先选择已知中文语音索引
                selected_voice = voices[12].id
            else:  # 按语言标识匹配
                for voice in voices:
                    for lang in voice.languages:
                        if isinstance(lang, bytes) and lang.decode('utf-8').startswith(('cmn', 'zh')):
                            selected_voice = voice.id
                            break
            if selected_voice:
                self.tts_engine.setProperty('voice', selected_voice)
                print(f"✅ 成功设置中文语音：{selected_voice}")
            else:
                print("⚠️ 未找到中文语音，可能导致播报异常")
            
            return True
        except Exception as e:
            print(f"❌ 离线TTS初始化失败: {e}")
            self.tts_engine = None
            return False

    def speak(self, text):
        """离线语音播报"""
        filtered_text = re.sub(r'[^\u4e00-\u9fa5，。？！,.;?!\s]', '', text).strip()
        if not filtered_text:
            print("⚠️ 过滤后无有效文本，无法播报")
            return
        def _speak_offline():
            if not self.tts_engine:
                print(f"⚠️ TTS未初始化，无法播报: {text}")
                return
            try:
                with self.tts_lock:  # 原有锁：防止同时播报
                    with self.audio_block_lock:  # 新增锁：阻断音频采集
                        self.tts_engine.say(filtered_text)
                        self.tts_engine.runAndWait()
            except Exception as e:
                print(f"❌ 离线播报错误: {e}")
        threading.Thread(target=_speak_offline, daemon=True).start()

    # ------------------------------
    # 任务点管理核心功能
    # ------------------------------
    def _load_task_points(self):
        """从文件加载任务点"""
        if os.path.exists(TASK_POINTS_FILE):
            try:
                with open(TASK_POINTS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载任务点失败: {e}")
        return {}  # 初始为空字典

    def _save_task_points(self):
        """保存任务点到文件（增加日志和容错）"""
        try:
            # 检查目录是否存在
            if not os.path.exists(os.path.dirname(TASK_POINTS_FILE)):
                os.makedirs(os.path.dirname(TASK_POINTS_FILE), exist_ok=True)
                print(f"📂 创建目录: {os.path.dirname(TASK_POINTS_FILE)}")
            
            # 写入文件
            with open(TASK_POINTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.task_points, f, ensure_ascii=False, indent=2)
            
            print(f"💾 任务点已保存至 {TASK_POINTS_FILE}（内容：{self.task_points}）")
            return True  # 保存成功
        except PermissionError:
            print(f"❌ 保存失败：无权限写入文件 {TASK_POINTS_FILE}")
            return False
        except Exception as e:
            print(f"❌ 保存失败：{e}（文件路径：{TASK_POINTS_FILE}）")
            return False
            
    def _get_scene_description(self, frame):
        """直接使用内存中的帧数据"""
        try:
            if frame is None:
                return "未获取到画面"
            
            # 直接转换帧为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # prompt = """识别图片中的核心物品（1-2个），遵循：
            # 1. 只说物体名称（如"打印机"）；
            # 2. 忽略颜色、位置等修饰；
            # 3. 若有多个，选最显眼的。
            # 十字以内完成。
            # """
            prompt = """识别图片中的核心物品（1-2个），遵循：
            1. 只说物体名称（如"打印机"）；
            2. 忽略颜色、位置等修饰；
            3. 若有多个，选最显眼的。
            十字以内完成。
            # """
            messages = [
                {"role": "observer", "content": [
                    {"type": "image", "image": pil_image},  # 直接使用PIL对象
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.qwen_processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.qwen_model.generate(** inputs, max_new_tokens=50)
            generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
            description = self.qwen_processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
            print(f"description: {description}")
            return description if description else "场景描述失败"
        except Exception as e:
            print(f"场景描述错误: {e}")
            return "场景识别失败"
            
    def add_task_point(self):
        """添加任务点（使用当前帧）"""
        if not self.current_pose:
            self.speak("未获取到底盘位置")
            return
        
        # 从当前位姿中提取x, y, yaw
        x, y, yaw = self.current_pose
        
        # 获取最新帧（保留队列数据）
        latest_frame = self._get_latest_frame()
        # 传递帧参数给场景描述方法
        scene_desc = self._get_scene_description(latest_frame)
        # 计算新任务点的ID：取现有任务点中最大的id+1，初始为0
        if self.task_points:
            # 提取所有现有任务点的id并转为整数，取最大值
            max_existing_id = max(int(point_info["id"]) for point_info in self.task_points.values())
            new_id = max_existing_id + 1
        else:
            # 若没有任务点，初始ID为0
            new_id = 0
        
        # 构建任务点完整信息（包含自增ID）
        task_point_info = {
            "id": new_id,  # 显式存储自增ID
            # 位姿数据（来自MQTT的base_status.pose）
            "pose": {
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(yaw, 3)
            },
            # 环境信息（模型识别的场景描述）
            "environment": scene_desc
        }
        
        # 以new_id为键，保存到任务点字典（键与数据中的id一致）
        self.task_points[str(new_id)] = task_point_info
        
        # 写入JSON文件（持久化存储）
        self._save_task_points()
        
        # 语音播报确认
        self.speak(
            f"已添加任务点{new_id}，"
            f"位置：x={x:.2f}, y={y:.2f}，"
            f"场景：{scene_desc}"
        )
        print(f"添加任务点{new_id}: {task_point_info}")


    def modify_task_point(self, point_id):
        """修改任务点（增加详细日志）"""
        print(f"收到指令：point_id={point_id}") 
        # 检查当前位姿
        if not self.current_pose:
            self.speak("未获取到底盘位置，无法修改任务点")
            print("❌ modify_task_point: current_pose为空")
            return
        
        # 检查任务点是否存在
        target_key = str(point_id)
        if target_key not in self.task_points:
            self.speak(f"未找到{point_id}号任务点")
            print(f"❌ modify_task_point: 任务点{point_id}不存在（当前任务点：{list(self.task_points.keys())}）")
            return
        
        # 打印修改前的数据
        print(f"修改前 - 任务点{point_id}: {self.task_points[target_key]}")
        
        # 获取新数据
        x, y, yaw = self.current_pose
        # 获取最新帧（保留队列数据）
        latest_frame = self._get_latest_frame()
        # 传递帧参数给场景描述方法
        scene_desc = self._get_scene_description(latest_frame)     
        print(f"新数据 - 位姿: (x={x}, y={y}, yaw={yaw}), 场景: {scene_desc}")
        
        # 更新内存中的任务点
        self.task_points[target_key] = {
            "id": int(point_id),
            "pose": {
                "x": round(x, 3),
                "y": round(y, 3),
                "yaw": round(yaw, 3)
            },
            "environment": scene_desc
        }
        
        # 打印修改后的数据
        print(f"修改后 - 任务点{point_id}: {self.task_points[target_key]}")
        
        # 保存到文件
        save_success = self._save_task_points()  # 修改_save_task_points使其返回是否成功
        if save_success:
            self.speak(f"已将当前位置设为{point_id}号点，场景：{scene_desc}")
        else:
            self.speak(f"修改任务点{point_id}失败，文件保存出错")

    
    
    # ------------------------------
    # MQTT相关回调
    # ------------------------------
    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"MQTT连接成功,状态码: {rc}")
        client.subscribe(self.mqtt_status_topic)  # 订阅底盘状态话题
        client.subscribe(self.mqtt_control_topic)  # 订阅控制指令话题
        client.subscribe("script/grab_status")   # 订阅机械臂抓取状态话题

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            
            # 仅处理来自 "script/grab_status" 话题的机械臂状态（循环导航中）
            if msg.topic == "script/grab_status" and "grab_successfully" in payload and self.is_cycling:
                arm_status = payload.get('grab_successfully', -1)
                current_point = self.cycle_order[self.current_cycle_step]  # 当前任务点（从循环顺序取）
                print(f"从 {msg.topic} 收到机械臂状态: {arm_status}（当前点：{current_point}，循环步骤：{self.current_cycle_step}）")

                # 状态码映射：机械臂状态→目标点ID（直接指定下一步要去的点）
                status_to_target = {
                    1: 2,  # 状态1→去2号点
                    2: 3,  # 状态2→去3号点
                    3: 0,  # 状态3→去0号点
                    0: None,  # 状态0→循环结束
                    -1: None  # 无效状态→不操作
                }

                # 获取目标点
                target_point = status_to_target.get(arm_status, None)

                # 处理不同状态场景
                if arm_status == -1:
                    self.speak("未获取到机械臂有效工作状态，循环暂停")
                    self.is_cycling = False  # 暂停循环
                    return

                elif arm_status == 0:
                    # 状态0：0号点完成（循环结束）
                    self.completed_cycles += 1
                    self.speak(f"已完成单次循环任务（1→2→3→0），共完成{self.completed_cycles}次")
                    self.is_cycling = False  # 终止循环
                    return

                elif target_point is not None:
                    # 计算下一步骤索引（根据目标点在循环顺序中的位置）
                    try:
                        next_step = self.cycle_order.index(target_point)
                        self.speak(f"机械臂在{current_point}号点工作完成，即将前往{target_point}号点（循环步骤{next_step+1}/4）")
                        
                        # 更新循环步骤计数器
                        self.current_cycle_step = next_step
                        # 导航到目标点
                        if str(target_point) in self.task_points:
                            self.send_goal_point(str(target_point))
                        else:
                            self.speak(f"循环中断：未找到{target_point}号任务点")
                            self.is_cycling = False
                    except ValueError:
                        self.speak(f"循环配置错误：目标点{target_point}不在循环顺序中")
                        self.is_cycling = False

                else:
                    # 未知状态
                    self.speak(f"收到未知机械臂状态{arm_status}，循环暂停")
                    self.is_cycling = False
                return
            
            # 处理底盘状态（更新当前位姿）
            if msg.topic == self.mqtt_status_topic and "pose" in payload:
                pose = payload["pose"]
                self.current_pose = (pose["x"], pose["y"], pose["yaw"])
                # print(f"更新当前位姿: x={pose['x']:.2f}, y={pose['y']:.2f}, yaw={pose['yaw']:.2f}")
                    
        except json.JSONDecodeError:
            print("MQTT消息解析失败")
            self.speak("指令解析错误，请重试")
        except Exception as e:
            print(f"处理MQTT消息错误: {e}")

        
    # ------------------------------
    # 命令处理
    # ------------------------------
    def process_image_recognition(self):
        """直接使用内存中的帧数据"""
        try:
            self.speak("正在识别，请稍候...")
            print("开始图像识别...")
            # 获取最新帧（不清空队列）
            latest_frame = self._get_latest_frame()

            if latest_frame is None:
                error_msg = "未获取到摄像头画面"
                self.speak(error_msg)
                print(error_msg)
                return
                
            # 直接转换帧为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(latest_frame, cv2.COLOR_BGR2RGB))
            # 在转换为PIL图像后添加resize
            target_size = (768, 768)  # 模型推荐尺寸
            pil_image = pil_image.resize(target_size, Image.LANCZOS)  # 高质量缩放

            prompt = """识别图片中的核心物品（1-2个），遵循：
            1. 只说物体名称；
            2. 忽略颜色、位置等修饰；
            3. 若有多个，选最显眼的。
            十字以内完成。
            """
            messages = [
                {"role": "observer", "content": [
                    {"type": "image", "image": pil_image},  # 直接使用PIL对象
                    {"type": "text", "text": prompt}
                ]}
            ]
            
            text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = self.qwen_processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.qwen_model.generate(** inputs, max_new_tokens=100)
            generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
            result = self.qwen_processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
            
            print(f"图像识别结果: {result}")
            # 提取数字并匹配任务点
            self.match_task_point_by_number(result)

            self.speak(f"识别到：{result}")
            self.speak("还有其他指令吗？")
            
        except Exception as e:
            print(f"图像识别错误: {e}")
            self.speak("识别失败，请重试")
        finally:
            torch.cuda.empty_cache()
    def extract_numbers_from_result(self, recognition_result: str) -> list:
        """增强版：支持复合数字、字母组合及动态映射"""
        if not isinstance(recognition_result, str):
            raise ValueError("输入必须是字符串类型")

        # 加载动态映射表（示例）
        self.char_to_id = {
            # 中文数字（0-9）
            "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
            # 字母（A-D/a-d → 0-3）
            "A": 0, "B": 1, "C": 2, "D": 3,
            "a": 0, "b": 1, "c": 2, "d": 3,
            "0": 0, "1": 1, "2": 2, "3": 3
        }
        task_ids = []

        # -------------------- 提取阿拉伯数字（支持复合数字） --------------------
        # 匹配连续数字（如"123"或"1号"中的"1"）
        arabic_pattern = r"\d+"
        arabic_matches = re.findall(arabic_pattern, recognition_result)
        for match in arabic_matches:
            task_ids.extend([int(d) for d in match])  # 支持多位数字拆分（如"123"→[1,2,3]）

        # -------------------- 提取中文数字（支持复合词） --------------------
        # 匹配中文数字（如"三号"→3）
        chinese_num_pattern = r"(零|一|二|三|四|五|六|七|八|九)+(号|点|层)?"
        chinese_matches = re.findall(chinese_num_pattern, recognition_result)
        for match in chinese_matches:
            num_str = match[0]
            if num_str in self.char_to_id:
                task_ids.append(self.char_to_id[num_str])

        # -------------------- 提取字母（支持大小写混合） --------------------
        # 匹配字母组合（如"A3"→13）
        letter_pattern = r"[A-Za-z0-9]+"
        letter_matches = re.findall(letter_pattern, recognition_result)
        for match in letter_matches:
            key = match[0].upper() if match else ""
            task_ids.append(self.char_to_id.get(key, -1))

        # -------------------- 过滤无效ID并去重 --------------------
        valid_ids = [tid for tid in set(task_ids) if str(tid) in self.task_points.keys()]
        return valid_ids
    
    def match_task_point_by_number(self, recognition_result: str):
        """添加调试信息"""
        if not isinstance(recognition_result, str):
            self.speak("识别结果格式异常")
            return

        task_ids = self.extract_numbers_from_result(recognition_result)
        print(f"【调试】提取的候选任务点ID: {task_ids}")  # 新增日志

        valid_points = []
        for tid in task_ids:
            point_key = str(tid)
            if point_key in self.task_points:
                valid_points.append((tid, self.task_points[point_key]))
            else:
                print(f"【警告】任务点ID {tid} 未在配置中定义")  # 新增日志

        if not valid_points:
            id_str = "、".join(map(str, task_ids))
            self.speak(f"未找到ID为 {id_str} 的任务点，请检查配置")
            return

        target_id, target_info = valid_points[0]
        print(f"【调试】匹配到任务点: {target_id} → {target_info}")  # 新增日志
        self.send_goal_point(str(target_id)) # 发送匹配到的任务点

    def process_dialogue(self, text):
        """处理对话请求"""
        try:
            self.speak("正在思考，请稍候...")
            print(f"处理对话: {text}")
            
            messages = [{"role": "friend", "content": [{"type": "text", "text": text}]}]
            text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to(self.device)
            
            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=200)
            generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
            response = self.qwen_processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
            
            print(f"对话回复: {response}")
            #对应josn环境描述

            # 检查对话内容是否涉及任务点环境描述，若涉及则导航
            self.check_dialogue_for_environment(text, response)

            self.speak(response)
        except Exception as e:
            print(f"对话处理错误: {e}")
            self.speak("处理对话时出错，请重试")

    def check_dialogue_for_environment(self, user_query, model_response):
        """
        使用千问模型分析对话内容，判断是否涉及任务点环境描述
        若涉及则查找对应任务点并发送导航指令
        """
        if not self.task_points:
            print("无任务点数据，跳过环境匹配")
            return
            
        # 准备当前所有任务点的环境描述，供模型参考
        environment_list = [f"ID {k}: {v['environment']}" for k, v in self.task_points.items()]
        environments_text = "\n".join(environment_list)
        
        # 构建提示词，让模型分析对话是否涉及任务点环境
        prompt = f"""请分析以下对话内容，判断是否涉及需要导航的环境地点：
        
        用户问：{user_query}
        回复：{model_response}
        
        现有可导航的环境地点列表：
        {environments_text}
        
        请按以下规则处理：
        1. 如果对话中明确提到了列表中的某个环境地点，请返回该地点的ID
        2. 如果提到的内容与某个环境地点高度相关，请返回该地点的ID
        3. 如果未提到任何环境地点或无法确定，请返回-1
        4. 只返回数字ID，不要返回任何额外文字
        5. 根据{user_query}{model_response}判断是否有符合对话内容要求的{environments_text}，返回其数字ID
        
        # 例如：
        # 用户问："打印机在哪里？"
        # 回复："打印机在3号点"
        # 列表中有"ID 3: 打印机"
        # 则返回：3
        """
        
        try:
            # 调用千问模型进行分析
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.qwen_processor(text=[text], padding=True, return_tensors="pt").to(self.device)
            
            generated_ids = self.qwen_model.generate(
                **inputs, 
                max_new_tokens=10,  # 只需要返回一个数字，限制输出长度
                temperature=0.1,    # 降低随机性，确保结果稳定
                top_p=0.9
            )
            
            generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
            result = self.qwen_processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
            
            # 解析模型返回的ID
            if result.isdigit():
                target_id = int(result)
                target_key = str(target_id)
                
                if target_key in self.task_points:
                    print(f"对话中识别到环境地点：ID {target_id}（{self.task_points[target_key]['environment']}）")
                    self.speak(f"检测到您提到了{self.task_points[target_key]['environment']}，是否需要导航过去？")
                    
                    # 直接发送检测到的点？ or 通过什么方式确认去点？
                    # 直接发送点
                    self.send_goal_point(target_key)
                else:
                    print(f"模型返回的ID {target_id} 不在任务点列表中")
            else:
                print(f"对话中未识别到需要导航的环境地点（模型返回：{result}）")
                
        except Exception as e:
            print(f"分析对话环境描述时出错：{e}")

    def send_lerobot(self, number):
        """向lerobot_status话题发送指定的阿拉伯数字"""
        # 根据传入的数字，执行对应机械臂动作如下 0～3：顺序对应起始点 一号点 二号点 三好点  9：打招呼  6：告知开始循环任务，机械臂工作完成给反馈
        try:
            # 确保传入的是有效的阿拉伯数字
            if not isinstance(number, int) or number < 0:
                raise ValueError(f"无效的数字: {number}，必须是非负整数")
            
            # 构造消息
            message = {
                "type": "greeting",
                "number": number
            }
            command_json = json.dumps(message)
            
            # 发布到lerobot_status话题（连续发3次确保送达）
            for _ in range(3):
                self.mqtt_client.publish(
                    topic="lerobot_status",  # 目标话题
                    payload=command_json,
                    qos=2  # 确保消息可靠传递
                )
                time.sleep(0.1)
            
            print(f"✅ 已向lerobot_status发送数字: {number}") # 发送信息，机械臂执行动作
            
            if number == 9:  # 仅打招呼任务播报
                self.speak(f"你好，我是小车，需要帮助吗")  # 指令问候播报问候语
        except Exception as e:
            print(f"❌ 发送打招呼消息失败: {e}")
            self.speak("打招呼失败，请重试")

    # ------------------------------
    # 导航功能（合并按描述导航）
    # ------------------------------
    def handle_navigation(self, command_code=None, query_text=""):
        """
        处理所有导航相关指令，包括：
        1. 按任务点ID导航（如去1号点）
        2. 按描述导航（如去拿水）
        """
        if command_code is not None and int(command_code) == 6:
            self.start_cycle_navigation()  # 启动循环导航
            return True
        if command_code is not None and int(command_code) == 7:
            self.is_cycling = False
            self.speak("已停止循环任务")
            return True
        
        # 尝试按环境描述匹配任务点
        matched_point = None
        if query_text:
            print(f"尝试按描述导航: {query_text}")
            matched_point = self.find_point_by_environment(query_text)
        
        # 如果找到匹配的任务点，导航到该点
        if matched_point:
            self.speak(f"找到匹配的位置，正在导航过去")
            success = self.send_goal_point(matched_point)
            if success:
                return True
        
        # 如果没有找到匹配，使用指定的任务点ID导航
        if command_code is not None:
            print(f"按任务点ID导航: {command_code}")

            # 确保任务点存在
            target_key = str(command_code)
            if target_key in self.task_points:
                self.send_goal_point(target_key)
                return True
            else:
                self.speak(f"未找到{command_code}号任务点，请先添加")
                return False

        # 如果没有指定任务点ID且没有找到匹配，提示用户
        self.speak("没有找到匹配的导航目标，请提供更明确的指令")
        return False

    # ------------------------------
    # 系统初始化与运行
    # ------------------------------
    def _get_camera_capture(self):
        """打开摄像头（优先选择外接USB摄像头）"""
        import glob
        import re
        camera_devices = glob.glob("/dev/video*")
        if not camera_devices:
            print("❌ 未检测到摄像头设备")
            return None
        
        # 排序设备：优先选择编号较高的设备（通常外接USB摄像头编号大于内置）
        camera_devices.sort(key=lambda x: int(re.findall(r'\d+', x)[0]), reverse=True)
        
        for dev_path in camera_devices:
            try:
                # 尝试打开设备，设置参数（适配USB摄像头常见分辨率）
                cap = cv2.VideoCapture(dev_path)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    # 验证是否能获取帧
                    ret, _ = cap.read()
                    if ret:
                        print(f"✅ 成功打开外接摄像头: {dev_path}")
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                print(f"尝试打开{dev_path}失败: {e}")
        
        print("❌ 所有摄像头设备均无法打开")
        return None
    
    def _get_latest_frame(self):
        """获取最新视频帧（不清空队列）"""
        latest_frame = None
        temp_frames = []
        # 取出所有帧并缓存
        while not self.video_queue.empty():
            temp_frames.append(self.video_queue.get())
        
        # 获取最后一帧
        if temp_frames:
            latest_frame = temp_frames[-1][0]
            # 将所有帧放回队列（保留数据）
            for frame, ts in temp_frames:
                self.video_queue.put((frame, ts))
        
        return latest_frame
    
    def initialize_system(self):
        try:
            if not self.initialize_tts():
                return False
            if not self.initialize_models():
                return False
            self.pyaudio_instance = pyaudio.PyAudio()
            self.audio_stream = self.pyaudio_instance.open(
                format=self.FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK
            )
            self.is_running = True
            return True
        except Exception as e:
            print(f"系统初始化失败: {e}")
            return False

    def start_listening(self):
        if not self.is_running:
            return

        self.speak(self.welcome_text)
        time.sleep(2)
        self.is_listening = True
        
        self.speak(self.listening_text)
        print("\n系统已准备就绪，请开始说话...")

        threading.Thread(target=self.audio_recorder, daemon=True).start()
        threading.Thread(target=self.video_recorder, daemon=True).start()
        threading.Thread(target=self._process_commands, daemon=True).start()

    def audio_recorder(self):
        """内存音频处理（无文件保存）"""
        self.last_active_time = time.time()
        audio_buffer = []

        while self.is_listening and self.is_running:
            # 检查是否正在播报（锁被占用），若是则跳过采集
            if self.audio_block_lock.locked():
                time.sleep(0.01)  # 短暂休眠避免忙等
                continue

            # 正常采集音频数据
            data = self.audio_stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            audio_buffer.append(data)

            if len(audio_buffer) * AUDIO_CHUNK / AUDIO_RATE >= 0.5:
                raw_audio = b''.join(audio_buffer)
                if self.check_vad_activity(raw_audio):
                    print("检测到语音活动")
                    self.last_active_time = time.time()
                    self.segments_to_save.append((raw_audio, time.time()))
                else:
                    print("静音中...")
                audio_buffer = []

            if time.time() - self.last_active_time > NO_SPEECH_THRESHOLD:
                if self.segments_to_save and self.segments_to_save[-1][1] > self.last_vad_end_time:
                    self.process_audio_in_memory()  # 改为内存处理
                    self.last_active_time = time.time()

    def process_audio_in_memory(self):
        """直接在内存中处理音频"""
        if not self.segments_to_save:
            return
            
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            
        # 获取关联的视频帧
        start_time = self.segments_to_save[0][1]
        end_time = self.segments_to_save[-1][1]
        video_frames = []
        while not self.video_queue.empty():
            frame, timestamp = self.video_queue.get()
            if start_time <= timestamp <= end_time:
                video_frames.append(frame)
        
        # 合并音频数据
        audio_data = b''.join([seg[0] for seg in self.segments_to_save])
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # 降采样到16kHz（语音识别模型要求）
        audio_16k = resample_poly(audio_np, 16000, AUDIO_RATE)
        audio_float32 = audio_16k.astype(np.float32) / 32768.0
        
        # 保存原始语音文本用于导航匹配
        self.last_nav_query = ""
        try:
            # 先进行语音识别以获取文本
            result = self.sensevoice_pipeline(audio_data, audio_fs=16000)
            raw_text = result[0].get("text", "").strip() if isinstance(result, list) else result.get("text", "")
            self.last_nav_query = re.sub(r'<\|.*?\|>|[^\u4e00-\u9fa5，。？！\s]', '', raw_text)
        except:
            pass
        
        # 启动推理线程
        threading.Thread(target=self.inference, args=(video_frames, audio_float32)).start()
        
        self.saved_intervals.append((start_time, end_time))
        self.segments_to_save.clear()

    def video_recorder(self):
        cap = self._get_camera_capture()
        if not cap:
            self.speak("摄像头初始化失败，请检查硬件连接")
            return
        
        print("视频录制已开始（外接摄像头）")
        while self.is_listening and self.is_running:
            ret, frame = cap.read()
            if ret:
                # 非阻塞方式写入队列，避免满队列阻塞
                if not self.video_queue.full():
                    self.video_queue.put((frame, time.time()), block=False)
                self.video_buffer.append((frame, time.time()))
                cv2.imshow("USB Camera Feed", frame)  # 视频窗口名 USB Camera Feed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("无法获取摄像头画面，重试中...")
                time.sleep(0.5)  # 减少重试频率，降低CPU占用

        cap.release()
        cv2.destroyAllWindows()
        print("视频录制已停止")

    # def check_vad_activity(self, audio_data):
    #     num, rate = 0, 0.6
    #     step = int(AUDIO_RATE * 0.02)
    #     flag_rate = round(rate * len(audio_data) // step)

    #     for i in range(0, len(audio_data), step):
    #         chunk = audio_data[i:i + step]
    #         if len(chunk) == step and self.vad.is_speech(chunk, AUDIO_RATE):
    #             num += 1
         
    #      # 增加能量检测
    #     audio_np = np.frombuffer(audio_data, dtype=np.int16)
    #     energy = np.sqrt(np.mean(audio_np**2))

    #     return num > flag_rate

    def check_vad_activity(self, audio_data):
        # ==================== 参数调整（核心优化点） ====================
        # VAD判定块数比例阈值（原0.4→0.6，需60%以上的块被判定为语音）
        THRESHOLD_RATIO = 0.5 
        # 音频能量绝对阈值（根据环境调试，值越大越严格，典型值1000~3000）
        MIN_ENERGY = 10      
        # 每块时长（20ms，48000Hz下每块960个采样点）
        STEP_MS = 0.02         
        step = int(AUDIO_RATE * STEP_MS)  # 计算每块的采样点数

        # ==================== 计算总块数 ====================
        total_samples = len(audio_data)
        total_chunks = total_samples // step  # 总块数（向下取整，避免不完整块）
        if total_chunks == 0:
            return False  # 音频数据不足一个块，无语音活动

        # ==================== 统计VAD语音块数 ====================
        speech_blocks = 0
        for i in range(total_chunks):
            start = i * step
            end = start + step
            chunk = audio_data[start:end]
            # 确保块长度正确（理论上不会发生，因total_chunks已取整）
            if len(chunk) != step:
                continue
            # 使用VAD判断当前块是否为语音（依赖已设置的VAD_MODE）
            if self.vad.is_speech(chunk, AUDIO_RATE):
                speech_blocks += 1

        # ==================== 计算能量 ====================
        # 转换为numpy数组计算均方根能量（反映整体音量）
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        energy = np.sqrt(np.mean(audio_np ** 2))  # 均方根能量
        # print(f"VAD块数比例: {speech_blocks} | 能量值: {energy} | 阈值: {MIN_ENERGY}")
        # ==================== 综合判断 ====================
        # 条件1：语音块数超过阈值比例（如60%）
        # 条件2：能量超过绝对阈值（过滤低音量噪声）
        return speech_blocks > int(total_chunks * THRESHOLD_RATIO) and energy > MIN_ENERGY
    
    def inference(self, video_frames, audio_data):
        """内存中推理（去除文件操作，指令分类与目标/导航需求解析）"""
        # 提取关键帧
        key_frames = []
        if video_frames:
            total_frames = len(video_frames)
            # 取3帧：前1/4、中间、后1/4（覆盖不同角度）
            frame_indices = [
                int(total_frames * 0.25),
                int(total_frames * 0.5),
                int(total_frames * 0.75)
            ]
            for idx in frame_indices:
                if 0 <= idx < total_frames:
                    frame = video_frames[idx]
                    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    key_frames.append(pil_frame)
        
        # 语音识别（直接从内存数据）
        try:
            result = self.sensevoice_pipeline(audio_data, audio_fs=16000)
            raw_text = result[0].get("text", "").strip() if isinstance(result, list) else result.get("text", "")
            raw_text = re.sub(r'<\|.*?\|>|[^\u4e00-\u9fa5，。？！\s]', '', raw_text)
            
            if not raw_text or len(raw_text) < 2:
                print(f"无效语音输入: {raw_text}")
                return
                
            print(f"语音识别结果: {raw_text}")
            # 保存原始文本用于导航匹配
            self.last_nav_query = raw_text

        except Exception as e:
            print(f"语音识别错误: {e}")
            self.speak("语音识别失败，请重试")
            return

        # 处理语音指令（融合分类与语义解析）
        try:
            prompt = (
                f"请分析以下语音指令，完成两项任务并严格按JSON格式输出：\n"
                f"语音指令：'{raw_text}'\n"
                f"任务1：指令分类\n"
                f" - 分类类型：导航、图像识别、对话、添加任务点、修改任务点、问候\n"
                f" - 分类规则：\n"
                f"   - 含'去'、'到'、'导航'、'拿'、'取'、'找'→导航；\n"
                f"   - 含'识别'、'看'、'检测'→图像识别；\n"
                f"   - 含'添加'、'增加'、'保存'、'记录'→添加任务点；\n"
                f"   - 含'修改'、'设定为'→修改任务点；\n"
                f"   - 含'问候'、'招呼'、'问好'→问候；\n"
                f"   - 无上述关键词→对话\n"
                f"   - 需提取分类依据的关键词（从原始语音中提取）\n"

                f"任务2：语义解析\n"
                f" - 提取target：用户想找/拿/取的核心物品（无则为空）；\n"
                f" - 判断need_navigate：是否需要导航（true/false）；\n"
                f" - 说明reason：分类及导航判断的综合依据\n"
                f"输出格式（仅JSON，无多余文本）：\n"
                f'{{"type": "指令类型", "keyword": "分类关键词", "target": "目标物品", "need_navigate": true/false, "reason": "综合判断依据"}}\n'
                f"示例：\n"
                f'指令："去拿桌子上的水瓶" → 输出：{{"type": "导航", "keyword": "去、拿", "target": "水瓶", "need_navigate": true, "reason": "含\'去、拿\'关键词属于导航，需拿水瓶故需要导航"}}\n'
                f'指令："识别一下这是什么" → 输出：{{"type": "图像识别", "keyword": "识别", "target": "", "need_navigate": false, "reason": "含\'识别\'关键词属于图像识别，与导航无关"}}\n'
                f'指令："今天天气如何" → 输出：{{"type": "对话", "keyword": "", "target": "", "need_navigate": false, "reason": "无特定关键词属于对话，与导航无关"}}'
            )

            # 构造千问模型的输入（包含关键帧辅助理解）
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            if key_frames:
                # 插入所有关键帧辅助场景理解
                for i, frame in enumerate(key_frames):
                    messages[0]["content"].insert(i, {"type": "image", "image": frame})

            # 调用千问模型解析
            text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages) if key_frames else (None, None)
            
            inputs = self.qwen_processor(
                text=[text], 
                images=image_inputs, 
                padding=True, 
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=200)
            generated_ids_trimmed = generated_ids[0][len(inputs['input_ids'][0]):]
            combined_result = self.qwen_processor.decode(generated_ids_trimmed, skip_special_tokens=True).strip()
            
            print(f"融合解析结果: {combined_result}")
            # 处理融合结果（整合原两种处理逻辑）
            self.process_combined_result(combined_result, raw_text)

        except Exception as e:
            print(f"指令解析错误: {e}")
            self.speak("未能理解您的指令，请重新表述")

    def process_combined_result(self, combined_result, raw_text):
        """
        处理融合后的指令指令解析结果
        combined_result: 模型返回的融合解析结果（JSON字符串）
        raw_text: 原始始语音识别文本
        """
        # 清理模型输出格式（移除代码块标记和无关字符）
        cleaned_result = re.sub(r'^```json\s*', '', combined_result)
        cleaned_result = re.sub(r'\s*```$', '', cleaned_result)
        # 补充添加对下划线_的支持，确保need_navigate等字段正确保留
        cleaned_result = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\{\}\"\:\,\s\_truefalse]', '', cleaned_result).strip()
        print(f"清理后的融合结果: {cleaned_result}")

        try:
            # 解析JSON并校验必要字段
            result = json.loads(cleaned_result)
            
            # 提取核心参数，增加默认值处理
            cmd_type = result.get("type", "").strip()
            keyword = result.get("keyword", "").strip()
            target = result.get("target", "").strip()
            need_navigate = result.get("need_navigate", False)
            reason = result.get("reason", "无说明")

            # 校验指令类型合法性，增加容错处理
            valid_types = ["导航", "图像识别", "对话", "添加任务点", "修改任务点", "问候"]
            if cmd_type not in valid_types:
                print(f"⚠️ 检测到无效指令类型: {cmd_type}，尝试自动推断...")
                
                # 自动推断类型
                if any(kw in raw_text for kw in ["去", "到", "导航", "拿", "取", "找"]):
                    cmd_type = "导航"
                elif any(kw in raw_text for kw in ["识别", "看", "检测"]):
                    cmd_type = "图像识别"
                elif any(kw in raw_text for kw in ["添加", "增加", "保存", "记录"]):
                    cmd_type = "添加任务点"
                elif any(kw in raw_text for kw in ["修改", "设定为"]):
                    cmd_type = "修改任务点"
                elif any(kw in raw_text for kw in ["问候", "招呼", "问好"]):
                    cmd_type = "问候"
                else:
                    cmd_type = "对话"
                    
                print(f"自动推断指令类型为: {cmd_type}")

            print(f"融合解析详情：类型={cmd_type}，关键词={keyword}，目标={target}，需导航={need_navigate}，依据={reason}")

        except (json.JSONDecodeError, ValueError) as e:
            print(f"融合结果解析错误: {e}，原始输出: {combined_result}")
            # 尝试直接根据原始文本推断
            print("尝试直接根据原始文本推断指令类型...")
            if any(kw in raw_text for kw in ["去", "到", "导航", "拿", "取", "找"]):
                cmd_type = "导航"
                target = raw_text
                need_navigate = True
            elif any(kw in raw_text for kw in ["识别", "看", "检测"]):
                cmd_type = "图像识别"
                target = ""
                need_navigate = False
            elif any(kw in raw_text for kw in ["添加", "增加", "保存", "记录"]):
                cmd_type = "添加任务点"
                target = ""
                need_navigate = False
            elif any(kw in raw_text for kw in ["修改", "设定为"]):
                cmd_type = "修改任务点"
                target = ""
                need_navigate = False
            elif any(kw in raw_text for kw in ["问候", "招呼", "问好"]):
                cmd_type = "问候"
                target = ""
                need_navigate = False
            else:
                cmd_type = "对话"
                target = raw_text
                need_navigate = False
                
            print(f"从原始文本推断：类型={cmd_type}，目标={target}")
            keyword = ""
            reason = "解析失败后从原始文本推断"

        # 整合类型修正逻辑（优先于模型判断）
        image_keywords = {"检测", "识别", "图像", "目标检测", "摄像头", "看", "分析"}
        has_image_kw = any(kw in raw_text for kw in image_keywords)
        if has_image_kw and cmd_type != "图像识别":
            print(f"强制修正类型：{cmd_type} → 图像识别（原始语音含图像关键词）")
            cmd_type = "图像识别"

        # 关键词直接匹配回退逻辑（优先级最高）
        for kw, cmd in self.command_mapping.items():
            if kw in raw_text:
                print(f"触发关键词直接匹配：{kw} → {cmd}")
                self.command_queue.put((cmd, kw))
                return

        # 按指令类型处理（整合分类与语义解析结果）
        try:
            if cmd_type == "导航":
                # 结合语义解析的target和need_navigate处理导航
                if need_navigate and target:
                    # 优先用解析出的target匹配任务点
                    print(f"尝试用目标物品'{target}'匹配任务点...")
                    matched_point = self.find_point_by_environment(target)
                    if matched_point:
                        # 直接使用匹配到的任务点ID导航，直接调用send_goal_point方法
                        print(f"找到匹配任务点: {matched_point}，直接导航")
                        # 增加任务点描述的语音反馈
                        desc = self.task_points.get(matched_point, {}).get("environment", "未知位置")
                        self.speak(f"正在导航到{target}所在位置（任务点{matched_point}，{desc}）")
                        # 直接调用函数发送目标点
                        self.send_goal_point(matched_point)
                        return
                    else:
                        self.speak(f"未找到包含'{target}'的任务点，将按原始指令搜索")
                
                # 未匹配到目标时，用原始文本导航
                self.command_queue.put((CommandType.NAVIGATE, raw_text))

            elif cmd_type == "图像识别":
                self.command_queue.put((CommandType.IMAGE_RECOGNITION, keyword or raw_text))

            elif cmd_type == "对话":
                # 结合target提供更精准的对话上下文
                dialogue_content = f"{target}相关内容：{raw_text}" if target else raw_text
                self.command_queue.put((CommandType.DIALOGUE, dialogue_content))

            elif cmd_type == "添加任务点":
                # 用target作为任务点描述（若有）
                point_desc = target if target else keyword
                self.command_queue.put((CommandType.ADD_TASK_POINT, point_desc))

            elif cmd_type == "修改任务点":
                # 从result提取point_id，默认0
                point_id = result.get("point_id", 0) if isinstance(result, dict) else 0
                # 用target作为修改后的描述（若有）
                new_desc = target if target else keyword
                self.command_queue.put(((CommandType.MODIFY_TASK_POINT, point_id), new_desc))

            elif cmd_type == "问候":
                self.command_queue.put((CommandType.GREET, 9))  # 9为问候指令参数

        except Exception as e:
            print(f"融合结果处理逻辑错误: {e}")
            self.speak("处理指令时出错，请重试")
    

    def _process_commands(self):
        """处理命令队列中的指令"""
        while self.is_running:
            try:
                command = self.command_queue.get(timeout=1)
                # 解析指令：可能是 (类型, 参数) 或 ((类型, 子参数), 参数)
                cmd_code, param = command[0], command[1] 
                # 处理元组类型的指令（如 MODIFY_TASK_POINT 包含子参数）
                if isinstance(cmd_code, tuple):
                    cmd_type, sub_param = cmd_code  # 拆分元组：(8, 0) → 类型8，子参数0
                    print(f"📌 解析到元组指令：类型={cmd_type}，子参数={sub_param}，参数={param}")
                else:
                    cmd_type = cmd_code  # 非元组类型，直接使用原始类型
                    sub_param = None  # 无额外子参数
                    print(f"解析到普通指令：类型={cmd_type}，参数={param}")
                
                self.speak(f"收到指令: {param}")

                # 用 cmd_type 进行判断
                if cmd_type == CommandType.ADD_TASK_POINT:
                    self.add_task_point()
                elif cmd_type == CommandType.MODIFY_TASK_POINT:
                    print(f"🔍 准备调用modify_task_point，参数point_id={sub_param}")
                    self.modify_task_point(sub_param)  # 调用修改任务点函数
                    print(f"modify_task_point调用完成，参数point_id={sub_param}")

                # elif cmd_type == CommandType.IMAGE_RECOGNITION:
                #     self.process_image_recognition()
                # elif cmd_type == CommandType.DIALOGUE:
                #     self.process_dialogue(param)
                # elif cmd_type == CommandType.GREET:
                #     self.send_lerobot(int(sub_param))
                # elif cmd_type == CommandType.NAVIGATE:
                #     # 处理导航指令，合并了按ID和按描述导航
                #     # 如果sub_param存在，说明是特定任务点ID导航
                #     # 否则使用param作为查询文本进行按描述导航
                #     self.handle_navigation(command_code=sub_param, query_text=param)
                else:
                    print(f"未知指令类型: {cmd_type}")
                    self.speak("未知指令，请重试")
                
                self.command_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"命令处理错误: {e}")
                self.speak("处理指令时出错，请重试")

    def stop_system(self):
            """完善的资源释放逻辑"""
            print("开始关闭系统...")
            # 停止标志
            self.is_listening = False
            self.is_running = False
            self.recording_active = False
            
            # 停止音频播放
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
            
            # 关闭MQTT连接
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            
            # 关闭音频流
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            
            # 终止PyAudio
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            # 等待线程结束
            if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2)
                print(f"音频线程已{'正常' if not self.audio_thread.is_alive() else '强制'}结束")
            
            if hasattr(self, 'video_thread') and self.video_thread.is_alive():
                self.video_thread.join(timeout=2)
                print(f"视频线程已{'正常' if not self.video_thread.is_alive() else '强制'}结束")
            
            if hasattr(self, 'command_thread') and self.command_thread.is_alive():
                self.command_thread.join(timeout=2)
                print(f"命令处理线程已{'正常' if not self.command_thread.is_alive() else '强制'}结束")
            
            # 清理CUDA内存
            if self.device and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            print("系统已完全关闭")


    def run(self):
        try:
            self.start_listening()
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.speak("收到退出信号，再见")
            time.sleep(1)
        finally:
            self.stop_system()

def main():
    nav_system = VoiceNavigationSystem()
    nav_system.run()

if __name__ == "__main__":
    main()

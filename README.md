# voic_interaction
# 项目说明

欢迎查看本项目！本项目包含与任务点及命令类型相关的代码文件，主要用于定义和处理各类任务指令，以下是对相关文件的简要说明：
移动式双臂协作机器人——语音识别与大模型应用
基于FunASR在线实时语音识别代码（github地址：https://github.com/ABexit/ASR-LLM-TTS）
进行二次开发，目的为实现离线语音识别、多模态大模型理解（语音+环境）以及对应底盘话题接口使用mqtt网络协议在同一局域网下完成导航部分的功能，创建josn文件存储位置点坐标、环境等信息，实现按描述内容导航到对应点、按说出的任务点前往对应点，以及添加修改josn中的位置点信息。使用千问2.0-VL模型完成图像检测以及长短对话功能。
   环境配置
    conda create -n llm python=3.10
conda activate llm

touch版本2.6.0+cu124
# CUDA 12.4 
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


pip install edge-tts==6.1.17 funasr==1.1.12 ffmpeg==1.4 opencv-python==4.10.0.84 transformers==4.45.2 webrtcvad==2.0.10 qwen-vl-utils==0.0.8 pygame==2.6.1 langid==1.1.6 langdetect==1.0.9 accelerate==0.33.0 PyAudio==0.2.14


    conda install -c conda-forge pynini=2.1.6
pip install WeTextProcessing --no-deps


pip install HyperPyYAML==1.2.2 modelscope==1.15.0 onnxruntime==1.19.2 openai-whisper==20231117 importlib_resources==6.4.5 sounddevice==0.5.1 matcha-tts==0.0.7.0



### 1. add_modify_point.py
该文件中定义了`CommandType`类，用于规范各类命令的类型。其中包含了多种命令类型的常量定义，如常规任务点之间的切换（A_TO_B、B_TO_C等）、图像检测（IMAGE_RECOGNITION）、对话类型（DIALOGUE）、添加任务点（ADD_TASK_POINT）、修改任务点（MODIFY_TASK_POINT）、导航（NAVIGATE）以及打招呼（GREET）等。这些命令类型常量为系统中不同任务的识别和处理提供了统一的标识。

### 2. test.py
此文件同样定义了`CommandType`类，其内容与`add_modify_point.py`中的`CommandType`类完全一致。该文件可能用于测试场景，确保在测试环境中能够正确引用和使用各类命令类型，与实际业务代码中的命令类型保持一致性，便于进行相关功能的测试验证。

## 使用说明
在项目中，您可以直接导入`CommandType`类来使用这些命令类型常量。例如，当需要处理添加任务点的操作时，可以使用`CommandType.ADD_TASK_POINT`来标识该命令类型，以便系统进行相应的处理。

## 注意事项
- 请确保在使用过程中，对`CommandType`类中的常量进行正确引用，避免因常量值错误导致的功能异常。
- 若需要新增命令类型，请在相关文件中统一添加并维护，确保各文件中`CommandType`类的一致性。

希望以上说明能帮助您更好地理解本项目的相关文件，如有任何问题，欢迎提出issues进行交流！

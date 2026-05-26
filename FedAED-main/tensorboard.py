import os
import subprocess
from pathlib import Path


def convert_mp4_to_wav_ffmpeg(input_folder: str, output_folder: str):
    # 如果输出文件夹不存在，则创建
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 遍历文件夹中的所有 mp4 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"{Path(filename).stem}.wav")

            print(f"正在转换: {input_file} -> {output_file}")

            try:
                # 使用 ffmpeg 转换音频格式
                subprocess.run([
                    "ffmpeg",
                    "-i", input_file,  # 输入文件
                    "-vn",  # 禁用视频
                    "-acodec", "pcm_s16le",  # 使用 pcm_s16le 音频编码（无损压缩）
                    "-ar", "16000",  # 设置音频采样率为 16kHz
                    "-ac", "1",  # 设置音频通道为单声道
                    output_file  # 输出文件
                ], check=True)

                print(f"转换成功: {output_file}")
            except subprocess.CalledProcessError as e:
                print(f"转换失败: {input_file} -> {output_file}, 错误: {e}")


# 使用示例
input_folder = "E:/fed-multimodal-main/fed-multimodal-main/fed_multimodal/data/meld/MELD.Raw/train_splits/waves"  # 替换为你的 MP4 文件夹路径
output_folder = "E:/fed-multimodal-main/fed-multimodal-main/fed_multimodal/data/meld/MELD.Raw/train_splits/waves"  # 替换为你希望保存 WAV 文件的文件夹路径

convert_mp4_to_wav_ffmpeg(input_folder, output_folder)

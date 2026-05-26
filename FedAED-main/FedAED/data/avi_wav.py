import os
import ffmpeg

def force_extract_audio(input_file, output_file):
    """强制提取音频，即使 FFmpeg 认为没有音频流"""
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, format="wav", acodec="pcm_s16le", ac=1, ar=16000)
            .run(overwrite_output=True, quiet=False)
        )
        print(f"[成功] 强制转换: {output_file}")
    except ffmpeg.Error as e:
        print(f"[失败] {input_file} 仍无法提取音频\n错误详情: {e.stderr.decode()}")

def batch_convert_avi_to_wav(root_folder):
    """遍历文件夹，将所有 AVI 文件转换为 WAV"""
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".avi"):
                avi_path = os.path.join(subdir, file)
                wav_path = os.path.join(subdir, file.replace(".avi", ".wav"))

                # 先尝试正常转换
                try:
                    (
                        ffmpeg
                        .input(avi_path)
                        .output(wav_path, format="wav", acodec="pcm_s16le", ac=1, ar=16000)
                        .run(overwrite_output=True, quiet=True)
                    )
                    print(f"[成功] {wav_path}")
                except ffmpeg.Error:
                    # 如果失败，则强制提取音频
                    print(f"[警告] 正常方法失败，尝试强制提取: {avi_path}")
                    force_extract_audio(avi_path, wav_path)

# 设置根目录
root_folder = r"E:\fed-multimodal-main\fed-multimodal-main\fed_multimodal\data\ucf101"
batch_convert_avi_to_wav(root_folder)

"""
视频转文字工具
用法:
  python transcribe.py                  → 批量处理 videos/ 里所有视频
  python transcribe.py 记单词.mp4       → 只处理指定视频
输出: output/ 文件夹里生成 视频名_文字.txt + 视频名_总结.txt
"""
import sys
import subprocess
import os
import whisper
from summarize import summarize_with_agent

# 修复 Windows 终端中文/emoji 显示问题
sys.stdout.reconfigure(encoding="utf-8")

# 把 FFmpeg 加入 PATH（Whisper 内部加载音频时也需要 ffmpeg）
_ffmpeg_dir = os.path.expandvars(
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin"
)
os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")


def extract_audio(video_path, audio_path="temp_audio.wav"):
    """
    第 1 步：用 FFmpeg 从视频中提取音频
    ffmpeg 参数解释（大白话）：
      -i     → 输入文件
      -vn    → 不要视频，只要音频
      -acodec pcm_s16le → 转成 WAV 格式（无压缩，Whisper 处理起来最顺手）
      -ar 16000 → 采样率 16000Hz（Whisper 最喜欢这个频率）
      -ac 1  → 单声道（不需要立体声，省空间）
      -y     → 如果已有同名文件，直接覆盖
    """
    ffmpeg_bin = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.2-full_build\bin\ffmpeg.exe"
    )
    subprocess.run([
        ffmpeg_bin, "-i", video_path, "-vn",
        "-acodec", "pcm_s16le", "-ar", "16000",
        "-ac", "1", audio_path, "-y"
    ], check=True)
    print(f"✅ 音频已提取: {audio_path}")
    return audio_path


def transcribe(video_path, model=None, model_size="medium", language="zh"):
    """
    第 2 步：视频 → 文字 主流程
    model_size 可选: tiny, small, medium, large
      - tiny   → 最快，准确率差，适合测试
      - small  → 英文还行，中文一般
      - medium → ★ 中文推荐！准确率明显提升
      - large  → 最准，但非常慢，需要好显卡
    """
    # 用视频名生成独立临时音频文件，防止批量处理时冲突
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_file = f"temp_audio_{video_name}.wav"

    # 2.1 提取音频
    extract_audio(video_path, audio_file)

    # 2.2 加载模型（如果外部已加载就用现成的，否则自己加载）
    if model is None:
        print(f"⏳ 正在加载 Whisper 模型 ({model_size})...")
        model = whisper.load_model(model_size)
        print("✅ 模型加载完成")
    else:
        print("✅ 使用已加载的模型")

    # 2.3 语音识别
    print("⏳ 正在识别语音...")
    result = model.transcribe(
        audio_file,
        language=language,
        initial_prompt="以下是普通话的简体中文内容。",
    )
    print("✅ 识别完成")

    # 2.4 清理临时音频文件
    os.remove(audio_file)
    print(f"🗑️  已清理临时文件: {audio_file}")

    return result["text"]


# 输入输出文件夹
VIDEO_DIR = "videos"
OUTPUT_DIR = "output"
# 支持的视频格式
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')

if __name__ == "__main__":
    # 自动建 output 文件夹
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 确定要处理的视频列表
    if len(sys.argv) >= 2:
        # 指定了文件名 → 只处理这一个
        video_names = [sys.argv[1]]
    else:
        # 没指定 → 扫描 videos/ 里所有视频
        all_files = os.listdir(VIDEO_DIR)
        video_names = [f for f in all_files if f.lower().endswith(VIDEO_EXTENSIONS)]
        if not video_names:
            print(f"❌ {VIDEO_DIR}/ 文件夹里没有视频文件！")
            print(f"   支持的格式: {', '.join(VIDEO_EXTENSIONS)}")
            sys.exit(1)
        print(f"📂 在 {VIDEO_DIR}/ 里找到 {len(video_names)} 个视频，开始批量处理...")

    # ★ 加载 Whisper 模型（只加载一次，所有视频共用）
    print("⏳ 正在加载 Whisper 模型 (medium)...")
    model = whisper.load_model("medium")
    print("✅ 模型加载完成\n")

    # 逐个处理视频
    for i, filename in enumerate(video_names, 1):
        video_path = os.path.join(VIDEO_DIR, filename)

        if not os.path.exists(video_path):
            print(f"⚠️ 跳过（文件不存在）: {video_path}")
            continue

        video_name = os.path.splitext(filename)[0]

        # 进度条
        if len(video_names) > 1:
            print(f"\n{'='*50}")
            print(f"🎬 [{i}/{len(video_names)}] 正在处理: {filename}")
            print(f"{'='*50}")
        else:
            print(f"\n🎬 正在处理: {filename}")

        try:
            # 语音转文字
            text = transcribe(video_path, model=model)

            # 输出结果
            print("\n" + "=" * 50)
            print("📝 识别结果：")
            print("=" * 50)
            print(text)
            print("=" * 50)

            # 保存完整文字
            output_file = os.path.join(OUTPUT_DIR, f"{video_name}_文字.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"💾 文字已保存到: {output_file}")

            # ★ AI 总结
            try:
                summary = summarize_with_agent(text)
                print("\n" + "=" * 50)
                print("🤖 AI 总结：")
                print("=" * 50)
                print(summary)
                print("=" * 50)

                summary_file = os.path.join(OUTPUT_DIR, f"{video_name}_总结.txt")
                with open(summary_file, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"💾 总结已保存到: {summary_file}")
            except ValueError as e:
                print(f"\n⚠️  跳过 AI 总结：{e}")
            except Exception as e:
                print(f"\n⚠️  AI 总结失败：{e}")

        except Exception as e:
            print(f"❌ 处理 {filename} 时出错: {e}")
            continue

    # 全部完成
    if len(video_names) > 1:
        print(f"\n🎉 全部完成！共处理 {len(video_names)} 个视频")
        print(f"   结果在 {OUTPUT_DIR}/ 文件夹里")

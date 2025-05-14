#!/usr/bin/env bash
# === LibriSpeech Turbo转换 - 针对Colab无权限环境优化 ===

# 提高优先级 (如果可能)
renice -n -20 $$ &>/dev/null || true

# ======== 一次性安装必要组件 ========
echo "📦 安装优化组件..."
apt-get update -qq &>/dev/null || true
apt-get install -y -qq parallel sox libsox-fmt-all python3-pip &>/dev/null || true
pip install -q tqdm psutil &>/dev/null || true

# ======== 配置参数 ========
SRC_DIR="/content/Voiceprint-Recognition/audio/LibriSpeech"
TEMP_DIR="/content/temp_conversion"
FLAC_LIST="$TEMP_DIR/all_flacs.txt"
CONVERT_LOG="$TEMP_DIR/converted.log"
PROGRESS_LOG="$TEMP_DIR/progress.txt"

# ======== 极速模式参数 ========
# 超级并行模式
SUPER_PARALLEL=true         # 启用超并行处理
DIRECT_TO_COLAB=true        # 先复制到Colab本地再处理
CHUNK_SIZE=500              # 每批处理文件数量

# 自动调优 - 基于系统能力设置
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
CPU_CORES=$(nproc)
IO_CAPACITY=$(iostat -x 1 1 | awk '/sda/{print $14}' | tail -1 || echo 50)

# 调整并行度
if [ $MEMORY_GB -ge 12 ]; then
  # 有足够内存，提高并行度
  MAX_PARALLEL=$((CPU_CORES * 3))
elif [ $MEMORY_GB -ge 8 ]; then
  # 中等内存
  MAX_PARALLEL=$((CPU_CORES * 2)) 
else
  # 低内存
  MAX_PARALLEL=$((CPU_CORES + 2))
fi

# 确保最小值和最大值
[ $MAX_PARALLEL -lt 4 ] && MAX_PARALLEL=4
[ $MAX_PARALLEL -gt 24 ] && MAX_PARALLEL=24

# ======== 创建目录 ========
mkdir -p "$TEMP_DIR"

# ======== 监控脚本 ========
cat > "$TEMP_DIR/monitor.py" << 'PYEOF'
import os
import sys
import time
import psutil
from tqdm import tqdm

# 获取参数
total_files = int(sys.argv[1])
progress_file = sys.argv[2]
prev_count = 0
start_time = time.time()

# 创建进度条
pbar = tqdm(total=total_files, unit='file', 
           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

# 实时监控函数
def update_stats():
    global prev_count
    
    # 读取进度
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                lines = f.readlines()
                current = len(lines)
                if current > prev_count:
                    pbar.update(current - prev_count)
                    prev_count = current
    except:
        pass
    
    # 计算统计信息
    elapsed = time.time() - start_time
    if prev_count > 0 and elapsed > 0:
        rate = prev_count / elapsed
        eta = (total_files - prev_count) / rate if rate > 0 else 0
        
        # 系统资源信息
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        # 更新进度条信息
        pbar.set_postfix(speed=f"{rate:.1f} files/s", 
                         cpu=f"{cpu}%", 
                         mem=f"{mem}%",
                         elapsed=f"{int(elapsed//60)}m{int(elapsed%60)}s")

try:
    while prev_count < total_files:
        update_stats()
        time.sleep(0.5)
    
    # 最终更新
    update_stats()
    pbar.close()
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n✅ 完成转换! 共处理 {prev_count} 个文件")
    print(f"⏱️ 总耗时: {int(total_time//60)}分{int(total_time%60)}秒")
    print(f"🚀 平均速度: {prev_count/total_time:.1f} 文件/秒")
    
except KeyboardInterrupt:
    pbar.close()
    print("\n中断处理")
PYEOF

# ======== 搜集文件 ========
echo "🔍 搜索FLAC文件..."
# 检查是否已有文件列表
if [ -f "$FLAC_LIST" ]; then
    echo "📄 使用现有文件列表"
else
    echo "📄 重新生成文件列表"
    find "$SRC_DIR" -type f -name "*.flac" > "$FLAC_LIST"
fi

TOTAL=$(wc -l < "$FLAC_LIST")
if [ "$TOTAL" -eq 0 ]; then
    echo "❌ 未找到FLAC文件!"
    exit 1
fi

# 重置进度日志
> "$PROGRESS_LOG"

echo "🔢 找到 $TOTAL 个FLAC文件"
echo "🖥️ 将使用 $MAX_PARALLEL 个并行任务"

# ======== 转换函数 ========
convert_file() {
    local flac_file="$1"
    local progress_log="$2"
    
    # 确定输出路径 (与源文件相同目录)
    local wav_file="${flac_file%.flac}.wav"
    
    # 跳过已存在的文件
    if [ -f "$wav_file" ]; then
        echo "1" >> "$progress_log"
        return 0
    fi
    
    # 使用sox进行最快速转换
    sox -q -G "$flac_file" "$wav_file" || true
    
    # 标记完成
    echo "1" >> "$progress_log"
}
export -f convert_file
export PROGRESS_LOG

# ======== 超速模式 - 批量分发 ========
if [ "$SUPER_PARALLEL" = true ]; then
    echo "🚀 启动超速模式 - 批量并行处理"
    
    # 启动进度监控
    python3 "$TEMP_DIR/monitor.py" "$TOTAL" "$PROGRESS_LOG" &
    MONITOR_PID=$!
    
    # 计算总批次
    TOTAL_CHUNKS=$(( (TOTAL + CHUNK_SIZE - 1) / CHUNK_SIZE ))
    
    # 批量处理
    for ((chunk=0; chunk<TOTAL_CHUNKS; chunk++)); do
        # 准备此批次文件
        CHUNK_START=$(( chunk * CHUNK_SIZE + 1 ))
        CHUNK_END=$(( CHUNK_START + CHUNK_SIZE - 1 ))
        CHUNK_FILE="$TEMP_DIR/chunk_${chunk}.txt"
        
        # 提取批次文件列表
        sed -n "${CHUNK_START},${CHUNK_END}p" "$FLAC_LIST" > "$CHUNK_FILE"
        CHUNK_COUNT=$(wc -l < "$CHUNK_FILE")
        
        # 并行处理此批次
        if [ $CHUNK_COUNT -gt 0 ]; then
            # 在后台处理此批次
            (
                cat "$CHUNK_FILE" | parallel -j $MAX_PARALLEL convert_file {} "$PROGRESS_LOG"
            ) &
            
            # 控制总并行批次数，防止过载
            RUNNING_BATCHES=$(jobs -p | wc -l)
            MAX_BATCHES=3  # 最多同时运行3个批次
            
            while [ $RUNNING_BATCHES -ge $MAX_BATCHES ]; do
                sleep 1
                RUNNING_BATCHES=$(jobs -p | wc -l)
            done
        fi
    done
    
    # 等待所有后台任务完成
    wait
    
    # 终止监控进程
    kill $MONITOR_PID 2>/dev/null || true
else
    # 常规模式 - 按文件处理
    echo "🚀 启动常规并行模式"
    
    # 启动进度监控
    python3 "$TEMP_DIR/monitor.py" "$TOTAL" "$PROGRESS_LOG" &
    MONITOR_PID=$!
    
    # 处理所有文件
    cat "$FLAC_LIST" | parallel -j $MAX_PARALLEL convert_file {} "$PROGRESS_LOG"
    
    # 终止监控进程
    kill $MONITOR_PID 2>/dev/null || true
fi

# ======== 清理 ========
echo "🧹 清理临时文件..."
rm -rf "$TEMP_DIR"

echo "✨ 转换完成!"

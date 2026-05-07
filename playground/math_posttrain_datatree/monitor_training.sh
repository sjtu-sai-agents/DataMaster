#!/bin/bash
# 训练进度实时监控脚本

CHECKPOINT_DIR="$1"

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Usage: $0 <checkpoint_directory>"
    echo ""
    echo "Example:"
    echo "  $0 /data/yaxindu/datascientist/DataScientistEvomaster2/runs/math_posttrain_datatree_20260408_055528/workspaces/task_0/artifacts/checkpoints/d28cad738e2648368ac53046832b001e"
    exit 1
fi

LOG_FILE="$CHECKPOINT_DIR/trainer_log.jsonl"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    echo ""
    echo "Training may not have started yet, or the path is incorrect."
    exit 1
fi

echo "========================================================================"
echo "🔍 Training Progress Monitor"
echo "========================================================================"
echo "Log file: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"
echo "========================================================================"
echo ""

# Clear screen function
clear_screen() {
    printf "\033c"
}

# Main monitoring loop
while true; do
    clear_screen

    echo "========================================================================"
    echo "🚀 Training Progress - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================================"

    # Get latest training status
    LATEST=$(tail -1 "$LOG_FILE" 2>/dev/null)

    if [ -z "$LATEST" ]; then
        echo "⏳ Waiting for training to start..."
    else
        # Parse JSON (requires python)
        python3 << EOF
import json
import sys

try:
    data = json.loads('''$LATEST''')
    current = data.get('current_steps', 0)
    total = data.get('total_steps', 1)
    loss = data.get('loss', 0)
    lr = data.get('lr', 0)
    epoch = data.get('epoch', 0)
    percentage = data.get('percentage', 0)
    elapsed = data.get('elapsed_time', 'N/A')
    remaining = data.get('remaining_time', 'N/A')

    # Progress bar
    bar_width = 50
    filled = int(bar_width * current / total)
    bar = '█' * filled + '░' * (bar_width - filled)

    print(f"📊 Progress: [{bar}] {percentage:.1f}%")
    print(f"")
    print(f"   Step:      {current:4d} / {total}")
    print(f"   Epoch:     {epoch:.4f}")
    print(f"   Loss:      {loss:.4f}")
    print(f"   LR:        {lr:.2e}")
    print(f"")
    print(f"   ⏱️  Elapsed:   {elapsed}")
    print(f"   ⏳ Remaining: {remaining}")

except Exception as e:
    print(f"❌ Error parsing log: {e}")
    print(f"Raw: {'''$LATEST'''[:200]}")
EOF
    fi

    echo ""
    echo "========================================================================"
    echo "💻 GPU Status"
    echo "========================================================================"

    # GPU utilization
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=',' read -r idx util mem_used mem_total; do
        printf "GPU %s: %3s%% util | %5s / %5s MB\n" "$idx" "$util" "$mem_used" "$mem_total"
    done

    echo ""
    echo "========================================================================"
    echo "Last 3 steps:"
    echo "========================================================================"
    tail -3 "$LOG_FILE" | python3 -c "
import json
import sys
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        print(f\"  Step {d.get('current_steps', '?'):3d}: loss={d.get('loss', 0):.4f} lr={d.get('lr', 0):.2e} {d.get('remaining_time', 'N/A')}\")
    except:
        pass
"

    echo ""
    echo "Press Ctrl+C to exit"

    # Update every 3 seconds
    sleep 3
done

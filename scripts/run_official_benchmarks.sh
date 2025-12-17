#!/bin/bash
# Official Benchmark Runner for metal-candle
#
# This script runs official performance benchmarks on controlled hardware
# for release validation and performance claim documentation.
#
# Usage:
#   ./scripts/run_official_benchmarks.sh [options]
#
# Options:
#   --quick      Run with fewer iterations (for testing the script)
#   --no-mlx     Skip MLX comparison benchmarks
#   --help       Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
RUNS=5
QUICK_MODE=false
SKIP_MLX=false
COOLDOWN_SECONDS=60

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            RUNS=2
            COOLDOWN_SECONDS=10
            shift
            ;;
        --no-mlx)
            SKIP_MLX=true
            shift
            ;;
        --help)
            head -n 15 "$0" | tail -n 13
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}       metal-candle Official Benchmark Runner${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're on macOS with Apple Silicon
if [[ $(uname) != "Darwin" ]]; then
    echo -e "${RED}❌ Error: This script must be run on macOS${NC}"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}⚠️  Warning: Not running on Apple Silicon (arm64)${NC}"
    echo -e "${YELLOW}   Benchmarks will still run but may not reflect Metal performance${NC}"
fi

# Environment information
echo -e "${BLUE}📊 Environment Information:${NC}"
echo "  OS:           $(sw_vers -productName) $(sw_vers -productVersion)"
echo "  Architecture: $(uname -m)"
echo "  Hostname:     $(hostname)"
echo "  Date:         $(date)"
echo "  Rust:         $(rustc --version)"
if command -v python3 &> /dev/null && ! $SKIP_MLX; then
    echo "  Python:       $(python3 --version)"
    if python3 -c "import mlx" 2>/dev/null; then
        echo "  MLX:          Installed"
    else
        echo -e "  MLX:          ${YELLOW}Not installed${NC}"
        SKIP_MLX=true
    fi
fi
echo ""

# Check battery status
if pmset -g batt | grep -q "Battery Power"; then
    echo -e "${YELLOW}⚠️  Warning: Running on battery power${NC}"
    echo -e "${YELLOW}   For consistent results, connect to power${NC}"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Pre-flight checks
echo -e "${BLUE}🔧 Pre-flight Checks:${NC}"

# Check for other high-CPU processes
HIGH_CPU_PROCS=$(ps aux | awk '$3 > 20.0 {print $11}' | grep -v "^ps$\|^awk$" | head -5 || true)
if [[ -n "$HIGH_CPU_PROCS" ]]; then
    echo -e "${YELLOW}⚠️  Warning: High-CPU processes detected:${NC}"
    echo "$HIGH_CPU_PROCS" | sed 's/^/     /'
    echo ""
    read -p "Kill these processes or continue? (k/c/N) " -n 1 -r
    echo
    case $REPLY in
        [Kk])
            killall Chrome Safari Slack Discord "Google Chrome" || true
            echo -e "${GREEN}✅ Killed common high-CPU apps${NC}"
            ;;
        [Cc])
            echo -e "${YELLOW}⚠️  Continuing with other processes running${NC}"
            ;;
        *)
            exit 1
            ;;
    esac
fi

# Check if already running benchmarks
if pgrep -x "cargo" > /dev/null; then
    echo -e "${RED}❌ Error: cargo is already running${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
echo ""

# Create results directory
RESULTS_DIR="benchmark_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo -e "${BLUE}📁 Results will be saved to: ${RESULTS_DIR}${NC}"
echo ""

# Cool down period
if ! $QUICK_MODE; then
    echo -e "${BLUE}🧊 Cooling down system...${NC}"
    echo "   Waiting ${COOLDOWN_SECONDS} seconds for system to stabilize"
    sleep $COOLDOWN_SECONDS
    echo -e "${GREEN}✅ System ready${NC}"
    echo ""
fi

# Function to run a single benchmark
run_benchmark() {
    local name=$1
    local run=$2
    local total=$3
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Running: ${name} (Run ${run}/${total})${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    local output_file="${RESULTS_DIR}/${name}_run${run}.txt"
    
    if cargo bench --bench "$name" --all-features 2>&1 | tee "$output_file"; then
        echo -e "${GREEN}✅ Completed: ${name} (Run ${run}/${total})${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed: ${name} (Run ${run}/${total})${NC}"
        return 1
    fi
}

# Main benchmark execution
echo -e "${BLUE}🚀 Starting Benchmark Runs (${RUNS} runs each)${NC}"
echo ""

# List of benchmarks to run (excluding mlx_comparison if MLX not available)
BENCHMARKS=(
    "fused_lora_bench"
    "lazy_vs_eager"
    "inference"
    "training"
)

if ! $SKIP_MLX; then
    BENCHMARKS+=("mlx_comparison")
fi

# Run each benchmark multiple times
for benchmark in "${BENCHMARKS[@]}"; do
    for ((i=1; i<=RUNS; i++)); do
        if ! run_benchmark "$benchmark" "$i" "$RUNS"; then
            echo -e "${RED}❌ Benchmark failed, aborting${NC}"
            exit 1
        fi
        
        # Cool down between runs (except last run)
        if [[ $i -lt $RUNS ]]; then
            echo ""
            echo -e "${BLUE}🧊 Cooling down ${COOLDOWN_SECONDS}s before next run...${NC}"
            sleep $COOLDOWN_SECONDS
        fi
    done
done

# Generate summary
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                    BENCHMARK COMPLETE${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Save environment info
cat > "${RESULTS_DIR}/environment.txt" << EOF
Benchmark Run Information
========================

Date: $(date)
Hostname: $(hostname)
OS: $(sw_vers -productName) $(sw_vers -productVersion)
Architecture: $(uname -m)
Rust: $(rustc --version)
Cargo: $(cargo --version)

Hardware:
$(system_profiler SPHardwareDataType)

Git Info:
Branch: $(git rev-parse --abbrev-ref HEAD)
Commit: $(git rev-parse HEAD)
Status: $(git status --short)

Configuration:
Runs per benchmark: ${RUNS}
Quick mode: ${QUICK_MODE}
MLX comparison: $([[ $SKIP_MLX == "true" ]] && echo "Skipped" || echo "Included")
EOF

echo -e "${GREEN}✅ All benchmarks completed successfully${NC}"
echo ""
echo -e "${BLUE}📊 Results saved to: ${RESULTS_DIR}/${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo "  1. Review results in ${RESULTS_DIR}/"
echo "  2. Calculate median values for each benchmark"
echo "  3. Update BENCHMARKS.md with new numbers"
echo "  4. Commit benchmark results:"
echo ""
echo "     git add ${RESULTS_DIR}/"
echo "     git commit -m \"docs: benchmark results for $(git describe --tags --always)\""
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Open results directory
if command -v open &> /dev/null; then
    open "$RESULTS_DIR"
fi






#!/bin/bash
# AnatCoder 数据下载脚本
# 用法: bash scripts/download_data.sh [--all | --minimal]

set -euo pipefail

usage() {
  echo "Usage: bash scripts/download_data.sh [--minimal | --all]"
  echo "  --minimal: 准备起步验证用目录（默认，建议 3 个 case）"
  echo "  --all:     准备完整实验目录（需要手动下载完整数据集）"
}

MODE="minimal"
if [[ $# -gt 1 ]]; then
  usage
  exit 1
fi

if [[ $# -eq 1 ]]; then
  case "$1" in
    --minimal)
      MODE="minimal"
      ;;
    --all)
      MODE="all"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 1
      ;;
  esac
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="$ROOT_DIR/data/raw"
TS_DIR="$RAW_DIR/totalsegmentator"
AMOS_DIR="$RAW_DIR/amos"

mkdir -p "$TS_DIR" "$AMOS_DIR/imagesTr" "$AMOS_DIR/labelsTr"

echo "=== AnatCoder Data Preparation Helper ==="
echo "Mode: $MODE"
echo "Raw data root: $RAW_DIR"
echo ""

ts_case_count=$(find "$TS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
amos_img_count=$(find "$AMOS_DIR/imagesTr" -type f \( -name "*.nii" -o -name "*.nii.gz" \) | wc -l | tr -d ' ')
amos_lbl_count=$(find "$AMOS_DIR/labelsTr" -type f \( -name "*.nii" -o -name "*.nii.gz" \) | wc -l | tr -d ' ')

echo "Current local status:"
echo "  TotalSegmentator cases: $ts_case_count"
echo "  AMOS imagesTr files:    $amos_img_count"
echo "  AMOS labelsTr files:    $amos_lbl_count"
echo ""

echo "[TotalSegmentator v2]"
echo "  DOI: https://doi.org/10.5281/zenodo.6802614"
echo "  Note: 数据下载通常需要账户/API/条款同意，直链不稳定，不建议硬编码。"
echo "  目录建议: data/raw/totalsegmentator/<case_id>/ct.nii.gz"
echo "            data/raw/totalsegmentator/<case_id>/segmentations/*.nii.gz"
echo ""
echo "[AMOS 2022]"
echo "  Website: https://amos22.grand-challenge.org/"
echo "  Note: 需要在 Grand Challenge 注册并同意使用条款后下载。"
echo "  目录建议: data/raw/amos/imagesTr/*.nii.gz"
echo "            data/raw/amos/labelsTr/*.nii.gz"
echo ""

if [[ "$MODE" == "minimal" ]]; then
  echo "=== Minimal mode (Week 1) ==="
  echo "建议准备 3 个 case 用于起步验证："
  echo "  1) TotalSegmentator 腹部 1 例"
  echo "  2) TotalSegmentator 胸部 1 例"
  echo "  3) AMOS 腹部 1 例"
  echo ""
  echo "完成后可运行："
  echo "  python scripts/preprocess_all.py --input_dir data/raw --output_dir data/processed --crop_size 128"
else
  echo "=== All mode (Full benchmark prep) ==="
  echo "请按官方流程下载完整数据并放入上述目录结构。"
  echo "建议先做最小集验证 pipeline，再扩展到全量。"
fi

echo ""
echo "Done. This script intentionally provides compliant instructions instead of direct dataset mirroring."

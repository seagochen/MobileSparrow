#!/usr/bin/env bash
set -u

# ===== Config (可通过参数覆盖) =====
DEST_DIR="./data/coco2017"
CONN=16             # aria2c -x 并发
SPLIT=16            # aria2c -s 分片
RETRIES=5           # 下载失败重试次数
WHAT="all"          # all | train | val | ann
SKIP_SPACE_CHECK=0  # 1=跳过磁盘空间检测
UNZIP_AFTER=1       # 1=下载后自动解压
LINK_INTO=""        # 在该目录下创建软链接到 annotations/train2017/val2017（例如: ./data）

TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# ===== Helpers =====
log()   { echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; }

usage() {
  cat <<EOF
Usage: $0 [--dest DIR] [--conn N] [--split N] [--retries N] [--what all|train|val|ann] [--no-unzip] [--skip-space-check] [--link-into DIR]

Options:
  --dest DIR            目标下载/解压目录（默认: ${DEST_DIR})
  --conn N              aria2c 并发连接数 -x（默认: ${CONN})
  --split N             aria2c 分片数 -s（默认: ${SPLIT})
  --retries N           下载失败重试次数（默认: ${RETRIES})
  --what WHAT           选择下载部分：all|train|val|ann（默认: ${WHAT})
  --no-unzip            仅下载不解压
  --skip-space-check    跳过磁盘空间检测
  --link-into DIR       在该目录下创建软链接到 annotations/train2017/val2017（例如: ./data）
  -h, --help            查看帮助
EOF
}

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "缺少依赖：$1"
    exit 1
  fi
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dest) DEST_DIR="$2"; shift 2 ;;
      --conn) CONN="$2"; shift 2 ;;
      --split) SPLIT="$2"; shift 2 ;;
      --retries) RETRIES="$2"; shift 2 ;;
      --what) WHAT="$2"; shift 2 ;;
      --no-unzip) UNZIP_AFTER=0; shift ;;
      --skip-space-check) SKIP_SPACE_CHECK=1; shift ;;
      --link-into) LINK_INTO="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) err "未知参数：$1"; usage; exit 1 ;;
    esac
  done
}

abspath()  { python3 - <<'PY' "$1"
import os,sys; print(os.path.abspath(sys.argv[1]))
PY
}
free_space_bytes() {
  df -Pk "$1" 2>/dev/null | awk 'NR==2{print $4*1024}'
}

check_space() {
  [[ "$SKIP_SPACE_CHECK" -eq 1 ]] && { warn "跳过磁盘空间检测"; return; }
  mkdir -p "$DEST_DIR_ABS"
  local need_zip=$(( (19 + 1 + 1) * 1024 * 1024 * 1024 )) # ~21GB 压缩包余量
  local need_unzip=$(( 45 * 1024 * 1024 * 1024 ))         # ~45GB 解压后建议
  local need_total=$(( need_zip + need_unzip ))
  local free=$(free_space_bytes "$DEST_DIR_ABS" || echo 0)

  if (( free < need_zip )); then
    warn "可用空间约 $(numfmt --to=iec --suffix=B $free)，可能不足以存放压缩包（建议≥$(numfmt --to=iec --suffix=B $need_zip)）。"
  fi
  if (( free < need_total )); then
    warn "可用空间约 $(numfmt --to=iec --suffix=B $free)，解压后可能不足（建议≥$(numfmt --to=iec --suffix=B $need_total)）。"
  fi
}

fname_from_url() { basename "$1"; }

download_and_verify() {
  local url="$1"
  local out="$2"  # 绝对路径
  local tries=0

  while : ; do
    tries=$((tries+1))
    log "下载（第 ${tries}/${RETRIES} 次）：$url"
    aria2c -c -x "$CONN" -s "$SPLIT" --file-allocation=none \
           -o "$(basename "$out")" -d "$(dirname "$out")" "$url" || true

    if [[ ! -f "$out" ]]; then
      warn "未找到文件：$out"
    else
      if unzip -tq "$out" >/dev/null 2>&1; then
        log "校验通过：$(basename "$out")"
        return 0
      else
        warn "校验失败：$(basename "$out")，将重试（断点续传）"
      fi
    fi

    if (( tries >= RETRIES )); then
      err "下载/校验多次失败：$url"
      return 1
    fi
    sleep 2
  done
}

maybe_unzip() {
  local zip_path="$1"     # 绝对路径
  local dest_abs="$2"     # 绝对路径
  [[ "$UNZIP_AFTER" -eq 0 ]] && { log "跳过解压：$zip_path"; return; }
  log "解压：$zip_path -> $dest_abs"
  mkdir -p "$dest_abs"
  unzip -q -n "$zip_path" -d "$dest_abs"
}

make_links() {
  local target_root="$1"    # 绝对路径
  local src_root="$2"       # 绝对路径（包含 train2017/ val2017/ annotations/）
  [[ -z "$target_root" ]] && return 0
  mkdir -p "$target_root"

  for d in annotations train2017 val2017; do
    local src="$src_root/$d"
    local dst="$target_root/$d"
    if [[ -e "$dst" || -L "$dst" ]]; then
      log "目标已存在：$dst (跳过覆盖)"
      continue
    fi
    if [[ -d "$src" ]]; then
      ln -s "$src" "$dst"
      log "已创建软链接：$dst -> $src"
    else
      warn "源目录不存在，无法链接：$src"
    fi
  done
}

main() {
  parse_args "$@"

  need aria2c
  need unzip

  DEST_DIR_ABS="$(abspath "$DEST_DIR")"
  mkdir -p "$DEST_DIR_ABS"
  check_space

  case "$WHAT" in
    all|train|val|ann) ;;
    *) err "非法 WHAT：$WHAT"; exit 1 ;;
  esac

  local fail=0
  local train_zip="$DEST_DIR_ABS/$(fname_from_url "$TRAIN_URL")"
  local val_zip="$DEST_DIR_ABS/$(fname_from_url "$VAL_URL")"
  local ann_zip="$DEST_DIR_ABS/$(fname_from_url "$ANN_URL")"

  if [[ "$WHAT" == "all" || "$WHAT" == "train" ]]; then
    download_and_verify "$TRAIN_URL" "$train_zip" || fail=1
    ((fail==0)) && maybe_unzip "$train_zip" "$DEST_DIR_ABS"
  fi
  if [[ "$WHAT" == "all" || "$WHAT" == "val" ]]; then
    download_and_verify "$VAL_URL" "$val_zip" || fail=1
    ((fail==0)) && maybe_unzip "$val_zip" "$DEST_DIR_ABS"
  fi
  if [[ "$WHAT" == "all" || "$WHAT" == "ann" ]]; then
    download_and_verify "$ANN_URL" "$ann_zip" || fail=1
    ((fail==0)) && maybe_unzip "$ann_zip" "$DEST_DIR_ABS"
  fi

  # 可选：创建软链接到指定根目录
  if [[ "$fail" -eq 0 && -n "$LINK_INTO" ]]; then
    local link_root_abs="$(abspath "$LINK_INTO")"
    make_links "$link_root_abs" "$DEST_DIR_ABS"
  fi

  if (( fail==0 )); then
    log "完成。数据位于：$DEST_DIR_ABS"
    [[ -n "$LINK_INTO" ]] && log "并已在 $LINK_INTO 下创建软链接。"
    exit 0
  else
    err "部分任务失败，稍后可再次执行继续断点续传。"
    exit 1
  fi
}

main "$@"

#!/bin/sh
# usage: status.sh <stage_no> <total> "<message>"   -> appends a line with a text progress bar to out/m12/STATUS.txt
S=~/Documents/AI/gemma4-qat/out/m12/STATUS.txt
N=$1; T=$2; shift 2
F=$((N*20/T)); BAR=$(printf '%*s' $F '' | tr ' ' '#')$(printf '%*s' $((20-F)) '' | tr ' ' '.')
echo "$(date '+%F %T %Z') [$BAR] $N/$T $*" >> $S

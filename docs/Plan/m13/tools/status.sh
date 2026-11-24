#!/bin/sh
S=~/Documents/AI/gemma4-qat/out/m13/STATUS.txt
N=$1; T=$2; shift 2
F=$((N*20/T)); BAR=$(printf '%*s' $F '' | tr ' ' '#')$(printf '%*s' $((20-F)) '' | tr ' ' '.')
echo "$(date '+%F %T %Z') [$BAR] $N/$T $*" >> $S

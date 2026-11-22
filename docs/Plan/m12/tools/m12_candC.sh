cd /Users/lexchien/Documents/AI/gemma4-qat
.venv-m0/bin/python docs/Plan/m12/tools/m12_vocab.py 114176 out/m12/work/vocab114176 > docs/Plan/m12/raw/vocab114176.log 2>&1
.venv-m0/bin/python docs/Plan/m12/tools/m12_vocab_check.py out/m12/work/vocab114176 > docs/Plan/m12/raw/vocab_check114176.log 2>&1
M12_Q4=ple_model_proj M12_EMB_BF16=1 docs/Plan/m12/tools/m12_make_trim.sh m8 out/m12/work/vocab114176
.venv-m0/bin/python docs/Plan/m12/tools/m12_native_eval.py Cm8 $PWD/out/m12/gemma4-e2b-m8-v114176-embbf16-q4_0.sbs $PWD/out/m12/bin/mac/gemma_native > docs/Plan/m12/raw/eval_Cm8.log 2>&1
RAW=docs/Plan/m12/raw .venv-m0/bin/python docs/Plan/m7/tools/m7_grade.py Cm8 > docs/Plan/m12/raw/grade_Cm8.log 2>&1
docs/Plan/m12/tools/status.sh 7 9 "stage7b 候選C eval done"
echo C_DONE

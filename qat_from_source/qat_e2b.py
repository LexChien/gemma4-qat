r"""Q4_0 QAT for Gemma 4 E2B with google-deepmind/gemma (kauldron config).
Based on actual repo APIs (commit 9e1f741, 2026-10-15):
  gm.nn.QuantizationAwareWrapper      gemma/gm/nn/_quantization.py
  peft.QuantizationMethod.Q4_0 (block 32, straight-through estimator)  gemma/peft/_quantization.py
  gm.ckpts.LoadCheckpoint(quantize=True) -> convert_to_qat_checkpoint  gemma/gm/ckpts/_checkpoint.py, _quantization.py
  gm.nn.Gemma4_E2B / gm.text.Gemma4Tokenizer   gemma/gm/nn/gemma4/_gemma4.py, gemma/gm/text/_tokenizer.py
  Trainer layout copied from examples/lora.py / examples/seq2seq.py
Run:
  python -m kauldron.main --cfg=qat_e2b.py --cfg.workdir=$PWD/workdir
Override ckpt:  --cfg.init_transform.path=/path/to/gemma4-e2b-it (Kaggle Flax / orbax dir)
UNVERIFIED: Q4_0 vs Q4_0_TRANSPOSE for FFN gating (see gemma/peft/README.md "Q4_0 assumes
transposed first FFN layer"); whether Gemma4_E2B accepts text_only=True (transformer reads it via getattr).
"""
from kauldron import konfig

with konfig.imports():
  from gemma import gm, peft
  from kauldron import kd
  import optax


def get_config():
  batch_size = 4       # Mac: keep 1-4
  max_length = 512
  return kd.train.Trainer(
      seed=42,
      train_ds=_ds(training=True, batch_size=batch_size, max_length=max_length),
      model=gm.nn.QuantizationAwareWrapper(
          method=peft.QuantizationMethod.Q4_0,
          model=gm.nn.Gemma4_E2B(tokens="batch.input", text_only=True),
      ),
      # quantize=True renames kernels into _SimulateQuantizedEinsum_* scopes
      init_transform=gm.ckpts.LoadCheckpoint(
          path=gm.ckpts.CheckpointPath.GEMMA4_E2B_IT,  # gs://gemma-data/checkpoints/gemma4-e2b-it
          quantize=True,
      ),
      num_train_steps=2_000,
      train_losses={
          "xentropy": kd.losses.SoftmaxCrossEntropyWithIntLabels(
              logits="preds.logits", labels="batch.target", mask="batch.loss_mask"),
      },
      optimizer=optax.adafactor(learning_rate=1e-5),  # full-param QAT, low LR
      checkpointer=kd.ckpts.Checkpointer(save_interval_steps=500),
  )


def _ds(*, training, batch_size, max_length):
  # Replace with your own data (JSON -> kd.data.py.Json / HF datasets); mtnt is the repo's demo set.
  return kd.data.py.Tfds(
      name="mtnt/en-fr", split="train" if training else "test",
      shuffle=training, num_epochs=None if training else 1,
      batch_size=batch_size, num_workers=2,
      transforms=[gm.data.Seq2SeqTask(
          in_prompt="src", in_response="dst",
          out_input="input", out_target="target", out_target_mask="loss_mask",
          tokenizer=gm.text.Gemma4Tokenizer(), max_length=max_length, truncate=True)],
  )

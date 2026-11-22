# 05 量化內部數學

> 路徑相對 `src/gemma/gemma/`。✅ 讀碼；⚠️ 推論/未執行。

## 1. 共同：STE（Straight-Through Estimator）
`_straight_through_estimator`（`peft/_quantization.py:316-333`）：
```
y = (x - stop_grad(x)) + stop_grad(Q(x))
```
前向值 = Q(x)，反向梯度 ∂y/∂x = 1（直接穿透）。`_simulate_uniform_quantization`（`:397`）與 `_simulate_sfp8_quantization`（`:474`）都被它包住；Q4_0 走 `_simulate_uniform_quantization` 的 PER_BLOCK 分支，所以也有 STE。✅

## 2. 各 QuantizationMethod（`simulate_quantize`，`:37-92`）
| method | 實作 | 數學 |
|---|---|---|
| NONE | 原樣 | x |
| INT4 | uniform, bit=4, PER_CHANNEL, `axis_to_reduce` 可指定 | s = 7 / max\|x\|（沿 reduce 軸，keepdims）；x̂ = clip(round(x·s), -8, 7) / max(s, 1e-6) |
| INT8 | uniform, bit=8, PER_CHANNEL（**忽略 axis_to_reduce**，`:70-75`） | s = 127 / max\|x\|（除最後一軸外全部 reduce，`reduce_max_all_but_one_axis(axis=-1)` → 每個輸出通道一個 scale） |
| Q4_0 | PER_BLOCK → `_q4_0(upper_bound=8, bitwidth=4, transpose=False)` | 見 §3 |
| Q4_0_TRANSPOSE | 同上 `transpose=True`（先 `'... a b -> ... b a'`） | 見 §3 |
| SFP8 | `_simulate_sfp8_quantization`（`:474-502`） | 每通道 scale = 1.875/max\|x\|；之後對 \|x\| 取 2 的冪次下界 lb，步長 lb/8（lb ≤ 1/256 時 lb/4），四捨五入；<1.12e-7 → 0，<1.49e-7 → 1.49e-7 |

PER_CHANNEL 在 `axis_to_reduce=None` 時：`reduce_max_all_but_one_axis(|x|, axis=-1)` → 只保留最後一軸 → **每個「最後一軸索引」一個 scale**（`peft/_quantization_utils.py:204-223`）。

## 3. Q4_0（`_q4_0`, `peft/_quantization.py:366-394`）
1. `_pack`（`:336-347`）：若 transpose 先交換最後兩軸；要求最後一軸可被 32 整除；`reshape(-1, 32)` 轉 f32 → **區塊沿最後一軸連續 32 個**。
2. 每塊（`:371-386`）：
```
m   = arr[argmax |arr|]           # 帶號的絕對值最大元素
d   = -m / 8                      # upper_bound=8
q   = clip(trunc(arr/d + 8.5), 0, 15)   # = round-half-up(arr/d) + 8
d16 = fp16(d)                     # scale 以 fp16 儲存模擬
x̂   = f32(d16) · (q - 8)          # 值域 {-8..7}·d，m 本身映射到 -8·d = m
```
3. `_unpack`（`:350-363`）：reshape 回原形、cast 回原 dtype、若 transpose 再轉回。
特性：16 級、非對稱（-8..7），最大值精確保留（除 fp16 舍入外），與 llama.cpp / GGML 的 Q4_0 公式相同（此處只作數學比對；本專案不使用 llama.cpp）。⚠️ 與官方 HF QAT 權重是否逐 bit 相同未驗證（04 §5 `frac_on_grid≈0.40`）。

## 4. 軸（axes）
- `get_axis_to_reduce_from_einsum_str`（`:505-521`）只認 Gemma 2/3 的 5 個字串：`BTD,NDH->BTNH`→(1,)、`BSD,CKDH->CBSKH`→(2,)、`BTNH,NHD->BTD`→(0,1)、`...F,NHF->...NH`→(2,)、`...H,HF->...F`→(0,)；其他 → None。
  - Gemma 4 的 PLE einsum（`...td,dnp->...tnp`、`...D,DP->...P`、`...P,PD->...D`）不在表中 → None。
  - ⚠️ bug 疑點：`SimulateQuantizedEinsum` 傳的是 `einsum_str=self.wrapped.name`（模組名，不是方程式，`:191-193`）→ 幾乎永遠回傳 None。只影響 `nn.Einsum` 路徑（INT4）。
- `_SimulateQuantizedEinsum`（gm 版，`gm/nn/_quantization.py:205-214`）呼叫 `peft.simulate_quantize(kernel, method)` **不給 axis** → INT4 為「每最後一軸一個 scale」，Q4_0 沿最後一軸分塊。
- 真量化 `peft.quantize`（`peft/_quantization_utils.py:65-157`）用 `_get_axis_to_reduce_from_weight_shape`（`:251-269`）以形狀猜層：`head_dim, d_model, _ = q_einsum.shape`（實際是 `N, D, H`，變數名誤標）。套到 Gemma 4 E2B：q[8,1536,H]→(1,)、attn_vec[8,H,1536]→(0,1)、gating[2,F,1536]→(2,)、linear 2D→(0,)、kv 4D→(2,)；**`per_layer_model_projection.w` [1536,35,256] 會落到 `raise ValueError('Unsupported weight shape')`**（shape[0]≠8、shape[2]≠1536）⚠️ 讀碼推論，未執行。另外 global 層 q [8,1536,512] 仍匹配。
- `peft.quantize` 只支援 NONE/INT4/INT8（`:89-101`），**沒有 Q4_0** ✅。

## 5. 哪些被包、哪些被跳過
| 元件 | QuantizationAwareWrapper | IntWrapper | LoRA | peft.quantize（離線） |
|---|---|---|---|---|
| `nn.Dense` / `nn.Einsum` | 包 | 包 | 包 | — |
| `gm.nn._layers.Einsum`（Gemma 2/3） | 包（`gm/nn/_quantization.py:136-147`） | 包（`:158-167`） | 包 | 有 `w` 的 dict 都量化 |
| `gm.nn.gemma4._layers.Einsum` / `ClippedEinsum` | **跳過**（class 不同） | **跳過** | 包（`gm/nn/_lora.py:37-38`） | 依 key `w` 量化（不看 class）⚠️ |
| `input_embedding`、`per_layer_embeddings`（`self.param` 直接建立，`_modules.py:74-78,101-105`） | 跳過（不是模組） | 跳過 | 跳過 | 跳過（無 `w` key） |
| RMSNorm scale、skip_scale | 跳過 | 跳過 | 跳過 | 跳過 |
| 視覺 / 音訊 encoder | 跳過（同上，class 不同）⚠️ | 跳過 ⚠️ | ClippedEinsum 被包 | 依 key |

→ 對 Gemma 4：repo QAT 在 **0 個** 層上做假量化（✅ 讀碼，⚠️ 未執行驗證）；即使修好 class 比對，嵌入表與 PLE 表（佔 2.75B 參數、bf16 5.5GB）仍不在 QAT 範圍，但官方 QAT ckpt 中它們**已在 Q4 格點上**（04 §5）。

## 6. 對「從原始碼重做 QAT」的實作需求（推導）
1. 在 `_replace_by_simulated_quantization` 加入 `gm.nn.gemma4._layers.Einsum`/`ClippedEinsum`，並保留 `w_scale`。
2. 依 04 §5 表格選軸：q/kv/attn_vec/linear/PLE gate/proj 用 transpose，gating 不 transpose（需要逐層 method，現行 wrapper 只有一個全域 `method`）。
3. 為 `input_embedding`、`per_layer_embeddings` 加入可微的 Q4_0 假量化（沿嵌入維分 32），對應官方做法。
4. 匯出：Q4_0 snap → JAX→HF 轉換（04 §3）→ MLX / compressed-tensors。

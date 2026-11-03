/-
  Spark.Kernel: Pure-Scalar (`RcMode.freestanding` / `LNIR`-Clean) Inference & GPU Arbitration Gates
  (`0 import Lean`, `0 sorryAx`)

  Unifies DGX Spark model-swap exclusivity, anti-thrashing `SWAP_GRACE_SEC` admission,
  background CUDA idle permit revocation (`chatBusy` + `asrBusy` + `swapActive`),
  and prompt-proportional prefill timeout budgets in a single pure Lean 4 kernel
  suitable for both Lake native linking and direct `LeanNative` `LNIR -> LLVM 18` static compilation.
-/

namespace Spark.Kernel

/--
  Unboxed C-ABI Background CUDA / Vanity GPU Permit Gate (`spark_gpu_permit_gate`):
  Returns `1` iff background GPU work may hold a short-lived permit, and `0` (revoke immediately)
  whenever:
  - `swapActive != 0` (cold model swap or relaunch in progress),
  - `chatBusy != 0` (foreground chat request queued, prefilling, or streaming on `:8001`),
  - `asrBusy != 0` (speech-to-text transcription in flight on `:8002`),
  - `operatorPaused != 0` (`/etc/spark-paused` or build lock held),
  - `routerAlive != 1` (router heartbeat missing or stale),
  - `idleElapsedMs < idleMinMs` (post-request cooldown window not yet elapsed), or
  - `freeMemMb < minFreeMemMb` (unified memory headroom below reserve, e.g. `< 12288 MiB`).
-/
@[inline, export spark_gpu_permit_gate]
def sparkGpuPermitGate (swapActive chatBusy asrBusy operatorPaused routerAlive : UInt8)
    (idleElapsedMs idleMinMs freeMemMb minFreeMemMb : UInt64) : UInt8 :=
  if swapActive == 0 && chatBusy == 0 && asrBusy == 0 && operatorPaused == 0 && routerAlive == 1 &&
     minFreeMemMb > 0 && freeMemMb ≥ minFreeMemMb && idleElapsedMs ≥ idleMinMs then 1
  else 0

/--
  Unboxed C-ABI Model-Swap & Anti-Thrash Admission Gate (`spark_swap_admit`):
  Determines how to route a request for `reqModelId` (`1..3`) given `curModelId` (`0` = none):
  - Returns `0` (` reject_invalid`): `reqModelId == 0` or `reqModelId > 3`.
  - Returns `1` (`reuse_ready`): `swapActive == 0 && backendHealthy == 1 && curModelId == reqModelId`.
  - Returns `2` (`launch_swap`): `swapActive == 0` and either `curModelId == 0`, `backendHealthy == 0`
    (crashed active model relaunch), or `(curModelId != reqModelId && idleSec ≥ graceSec)`.
  - Returns `3` (`refuse_busy_503`): `swapActive != 0` OR `(curModelId != 0 && curModelId != reqModelId && idleSec < graceSec)`.
    Prevents concurrent duplicate `swap-model.sh` launches and blocks ~15-minute model-swap ping-pong!
-/
@[inline, export spark_swap_admit]
def sparkSwapAdmit (curModelId reqModelId swapActive backendHealthy : UInt8)
    (idleSec graceSec : UInt64) : UInt8 :=
  if reqModelId == 0 || reqModelId > 3 then 0
  else if swapActive != 0 then 3
  else if curModelId == reqModelId && backendHealthy == 1 then 1
  else if curModelId == 0 || backendHealthy == 0 || idleSec ≥ graceSec then 2
  else 3

/--
  Unboxed C-ABI Retry-After Seconds Calculator (`spark_swap_retry_after_sec`):
  Computes the exact `Retry-After` header (`≥ 1`) when `spark_swap_admit == 3`.
-/
@[inline, export spark_swap_retry_after_sec]
def sparkSwapRetryAfterSec (swapActive : UInt8) (idleSec graceSec defaultSwapWaitSec : UInt64) : UInt64 :=
  if swapActive != 0 then
    if defaultSwapWaitSec == 0 then 30 else defaultSwapWaitSec
  else if graceSec > idleSec then
    graceSec - idleSec
  else 1

/--
  Unboxed C-ABI Prompt-Proportional Prefill Deadline (`spark_prefill_budget_ms`):
  Computes the prefill first-token deadline (`[basePrefillMs, maxPrefillMs]`) for a prompt of
  `promptBytes` UTF-8 bytes (`basePrefillMs + (promptBytes / 4096) * per4kBytesMs`, clamped to
  `maxPrefillMs` without `UInt64` wraparound), ensuring 150k-token agent prompts get sufficient
  prefill headroom while post-first-byte SSE idle checks stay strict.
-/
@[inline, export spark_prefill_budget_ms]
def sparkPrefillBudgetMs (promptBytes basePrefillMs per4kBytesMs maxPrefillMs : UInt64) : UInt64 :=
  if maxPrefillMs ≤ basePrefillMs then basePrefillMs
  else if per4kBytesMs == 0 || (promptBytes >>> 12) == 0 then basePrefillMs
  else if (promptBytes >>> 12) > (maxPrefillMs - basePrefillMs) / per4kBytesMs then maxPrefillMs
  else basePrefillMs + (promptBytes >>> 12) * per4kBytesMs

/--
  Unboxed C-ABI Trusted Mesh / Bearer Auth Gate (`spark_auth_gate`):
  Enforces `ROUTER_AUTH` (`authMode`: `0` = `off`, `1` = `external`, `2` = `all`):
  - Open health path (`isOpenPath == 1`) or `authMode == 0` -> `1`.
  - Direct loopback (`isDirectLoopback == 1 && viaProxy == 0`) -> `1`.
  - Under `external` (`authMode == 1`), trusted `lean-mesh` (`10.88.0.0/16`) or Tailscale peer (`isTrustedMesh == 1`) -> `1`.
  - Otherwise requires constant-time verified API key (`hasValidKey == 1`).
-/
@[inline, export spark_auth_gate]
def sparkAuthGate (authMode isOpenPath isDirectLoopback viaProxy isTrustedMesh hasValidKey : UInt8) : UInt8 :=
  if isOpenPath == 1 || authMode == 0 then 1
  else if isDirectLoopback == 1 && viaProxy == 0 then 1
  else if authMode == 1 && isTrustedMesh == 1 then 1
  else if hasValidKey == 1 then 1
  else 0

end Spark.Kernel

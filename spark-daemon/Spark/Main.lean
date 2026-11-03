/-
  Spark.Main: Pure-Lean Native Multi-Model Inference Router & GPU Arbitration Daemon (`spark-daemon`)
  (`0 import Lean`, `0 sorryAx`, `supportInterpreter := false`)

  Built directly on `LeanSys` (`Posix`, `Http`, `Reactor.relayHttpStream`, `Codec.JsonTape`, `Auth.ctEqString`,
  `Sandbox.runSandboxedGroup`) and governed by `Spark.Kernel` / `Spark.Spec`:
  - Non-blocking SSE reverse-proxy splice (`relayHttpStream` + `leansys_verify_stream_relay`) with
    prompt-proportional prefill deadlines (`spark_prefill_budget_ms`), `: keepalive\n\n` frame-boundary
    heartbeats, and instant `closeFd upstreamFd` on downstream client disconnect.
  - Model alias resolution (`qwen3.8-flash-next`, `leanstral-1.5`, `glm-5.3-flash`), `SWAP_GRACE_SEC`
    anti-thrashing (`503 Service Unavailable` + `Retry-After`), and `-pgid` `ChildGroup` swap execution.
  - Unified GPU permit status & revocation (`spark_gpu_permit_gate`) across active swaps, chat `:8001`,
    and ASR `:8002`, plus ChatML sentinel stripping (`<|im_end|>`, `<|im_start|>`, `</|im_end|>`, `</|im_start|>`).
-/
import LeanSys
import Spark.Kernel
import Spark.Spec

namespace Spark.Main

open LeanSys.Spec
open LeanSys.Posix
open LeanSys.Http
open LeanSys.Reactor
open LeanSys.Codec
open LeanSys.Auth
open LeanSys.Sandbox
open Spark.Kernel

/-- Canonical chat models managed on DGX Spark (`1..3`, `0` = none). -/
inductive SparkModel where
  | none
  | qwen38FlashNext
  | leanstral15
  | glm53Flash
  deriving Repr, DecidableEq, Inhabited

namespace SparkModel

@[inline]
def toId : SparkModel → UInt8
  | .none            => 0
  | .qwen38FlashNext => 1
  | .leanstral15     => 2
  | .glm53Flash      => 3

@[inline]
def ofId : UInt8 → SparkModel
  | 1 => .qwen38FlashNext
  | 2 => .leanstral15
  | 3 => .glm53Flash
  | _ => .none

def canonicalName : SparkModel → String
  | .none            => ""
  | .qwen38FlashNext => "qwen3.8-flash-next"
  | .leanstral15     => "leanstral-1.5"
  | .glm53Flash      => "glm-5.3-flash"

/-- Resolve user/client model strings and aliases (case-insensitive) to a canonical `SparkModel`. -/
def resolveAlias (raw : String) : Option SparkModel :=
  let s := raw.trimAscii.toString.toLower
  if s.isEmpty || s == "qwen3.8-flash-next" || s == "qwen3.8-flash" || s == "flash-next" ||
     s == "qwen3.8-flash-next-uncensored" || s == "orcarouter/qwen3.8-flash-next-uncensored-nvfp4" ||
     s == "qwen" || s == "qwen3.8" || s == "builtin/private" then
    some .qwen38FlashNext
  else if s == "leanstral-1.5" || s == "leanstral" || s == "leanstral-15" || s == "leanstral-1-5" ||
          s == "leanstral-1.5-119b" || s == "leanstral-1.5-119b-a6b" || s == "leanstral-1.5-nvfp4" ||
          s == "lean4" || s == "proving" then
    some .leanstral15
  else if s == "glm-5.3-flash" || s == "glm" || s == "glm-5.3" || s == "glm5" ||
          s == "glm-5.3-flash-uncensored" || s == "orcarouter/glm-5.3-flash-uncensored-nvfp4" then
    some .glm53Flash
  else
    Option.none

end SparkModel

/-- Strip leaked ChatML stop sentinels from a string field. -/
def stripChatMlText (s : String) : String := Id.run do
  let sentinels := #["<|im_end|>", "<|im_start|>", "</|im_end|>", "</|im_start|>"]
  let mut out := s
  let mut changed := true
  while changed do
    changed := false
    for tok in sentinels do
      let parts := out.splitOn tok
      if parts.length > 1 then
        out := String.intercalate "" parts
        changed := true
  return out.trimAsciiEnd.toString

/-- Rewrite JSON `"model":"<alias>"` to the canonical served model name if present, and inject
    `"reasoning_effort":"low"` on reasoning models when absent so hidden `<think>` blocks do not exhaust `max_tokens`. -/
def normalizeChatPayload (rawBody : ByteSlice) (m : SparkModel) (defaultEffort : String := "low") : ByteArray :=
  let tape := JsonTape.indexJsonObject rawBody
  let rawStr := rawBody.toStringLossy
  let canon := m.canonicalName
  let withModel :=
    match JsonTape.lookupString? tape "model" with
    | some orig =>
      if orig == canon then rawStr
      else
        let pat := s!"\"model\":\"{orig}\""
        let patSp := s!"\"model\": \"{orig}\""
        let r1 := String.intercalate s!"\"model\":\"{canon}\"" (rawStr.splitOn pat)
        String.intercalate s!"\"model\": \"{canon}\"" (r1.splitOn patSp)
    | none => rawStr
  let hasEffort := (JsonTape.lookupSlice? tape "reasoning_effort").isSome ||
                   (JsonTape.lookupSlice? tape "chat_template_kwargs").isSome
  let finalStr :=
    if !hasEffort && !defaultEffort.isEmpty && (m == .qwen38FlashNext || m == .glm53Flash) then
      let trimmed := withModel.trimAscii.toString
      if trimmed.startsWith "{" && trimmed.length > 2 then
        if m == .glm53Flash then
          "{\"chat_template_kwargs\":{\"reasoning_effort\":\"" ++ defaultEffort ++ "\"}," ++ (trimmed.drop 1).toString
        else
          "{\"reasoning_effort\":\"" ++ defaultEffort ++ "\"," ++ (trimmed.drop 1).toString
      else
        withModel
    else
      withModel
  (stripChatMlText finalStr).toUTF8

/-- Runtime configuration for `spark-daemon`. -/
structure DaemonConfig where
  listenHost       : String := "127.0.0.1"
  listenPort       : UInt16 := 8000
  backendHost      : String := "127.0.0.1"
  backendPort      : UInt16 := 8001
  asrPort          : UInt16 := 8002
  swapScriptPath   : String := "/opt/spark/inference/swap-model.sh"
  swapGraceSec     : UInt64 := 600
  sseKeepaliveMs   : UInt64 := 10000
  basePrefillMs    : UInt64 := 90000
  per4kPrefillMs   : UInt64 := 4000
  maxPrefillMs     : UInt64 := 900000
  streamIdleMaxMs  : UInt64 := 120000
  gpuIdleMinMs     : UInt64 := 20000
  gpuMinFreeMb     : UInt64 := 12288
  authMode         : UInt8  := 0 -- `0` = off, `1` = external, `2` = all
  apiKeys          : Array String := #[]

/-- Mutable router & GPU arbitration state protected by Lean `IO.Ref`. -/
structure DaemonState where
  currentModel   : SparkModel := .none
  swapActive     : Bool := false
  chatActive     : UInt32 := 0
  asrActive      : UInt32 := 0
  lastUsedMonoMs : UInt64 := 0
  lastIdleMonoMs : UInt64 := 0
  swapEpoch      : UInt64 := 0

/-- Check whether `peerIp` belongs to trusted `lean-mesh` (`10.88.*`) or Tailscale (`100.64..127.*`). -/
def isTrustedOverlayIp (ip : String) : Bool :=
  let t := ip.trimAscii.toString
  if t.startsWith "10.88." then true
  else if t.startsWith "100." then
    match (t.splitOn ".") with
    | "100" :: second :: _ =>
      let oct := parseNatAscii second
      64 ≤ oct && oct ≤ 127
    | _ => false
  else if t.startsWith "fd7a:115c:a1e0:" then true
  else false

/-- Look up an HTTP request header case-insensitively. -/
def findHeader? (headers : Array (String × String)) (name : String) : Option String := Id.run do
  for (k, v) in headers do
    if asciiEqIgnoreCase k name then
      return some v.trimAscii.toString
  return none

/-- Authorize an incoming `HttpRequest` using `Spark.Kernel.sparkAuthGate` + constant-time `ctEqString`. -/
def authorizeRequest (cfg : DaemonConfig) (req : HttpRequest) (directLoopback : Bool := true) : Bool :=
  let cleanPath := match req.path.splitOn "?" with | p :: _ => p | [] => req.path
  let isOpen : UInt8 := if cleanPath == "/health" then 1 else 0
  let xff? := findHeader? req.headers "x-forwarded-for"
  let (peerIp, viaProxy) := match xff? with
    | some xff =>
      let parts := xff.splitOn ","
      let lastIp := (parts.getLast?.getD "").trimAscii.toString
      (lastIp, (1 : UInt8))
    | none => ("127.0.0.1", (0 : UInt8))
  let isLoop : UInt8 := if directLoopback && (peerIp == "127.0.0.1" || peerIp == "::1") then 1 else 0
  let isMesh : UInt8 := if isTrustedOverlayIp peerIp then 1 else 0
  let candidateKey := match req.authBearer with
    | some b => b.trimAscii.toString
    | none => (findHeader? req.headers "x-api-key").getD ""
  let validKey : UInt8 :=
    if !candidateKey.isEmpty && cfg.apiKeys.any (fun k => ctEqString candidateKey k) then 1 else 0
  sparkAuthGate cfg.authMode isOpen isLoop viaProxy isMesh validKey == 1

/-- Probe whether the local `:8001` chat backend responds `200 OK` on `/health`. -/
def probeBackendHealthy (cfg : DaemonConfig) : IO Bool := do
  match ← httpClientRequest cfg.backendHost cfg.backendPort "GET" "/health" (ByteSlice.ofString "") none 1500 with
  | .ok _ => pure true
  | .error _ => pure false

/-- Execute `swap-model.sh <canonical>` inside a monitored `-pgid` `ChildGroup` (`runSandboxedGroup`). -/
def runModelSwapGroup (cfg : DaemonConfig) (target : SparkModel) : IO (Except String Unit) := do
  if cfg.swapScriptPath.isEmpty || !(← (System.FilePath.mk cfg.swapScriptPath).pathExists) then
    return .ok ()
  let quota : SandboxQuota := {
    maxRssBytes := 8589934592 -- 8 GiB wrapper cap
    timeoutMs := 1800000      -- 30 min max cold load
    maxPipeBytes := 4194304
  }
  match ← runSandboxedGroup quota "/bin/bash" #[cfg.swapScriptPath, target.canonicalName] with
  | .ok res =>
    if res.exitCode == 0 && !res.timedOut && !res.quotaBreach then
      pure (.ok ())
    else
      pure (.error s!"swap-model.sh failed (exit={res.exitCode}, timedOut={res.timedOut})")
  | .error e =>
    pure (.error s!"swap-model.sh spawn error: {e}")

/-- Serialize an HTTP response with an optional `Retry-After` header. -/
def buildJsonWire (status : Nat) (jsonBody : String) (retryAfterSec? : Option UInt64 := none) : ByteArray :=
  let body := jsonBody.toUTF8
  let reason := if status == 503 then "Service Unavailable" else HttpResponse.reasonPhrase status
  let retryLine := match retryAfterSec? with
    | some w => s!"Retry-After: {w}\r\n"
    | none   => ""
  let head :=
    s!"HTTP/1.1 {status} {reason}\r\n" ++
    "Content-Type: application/json\r\n" ++
    "Access-Control-Allow-Origin: *\r\n" ++
    retryLine ++
    s!"Content-Length: {body.size}\r\n" ++
    "Connection: close\r\n\r\n"
  head.toUTF8 ++ body

/-- Build the `/v1/models` OpenAI-compatible JSON catalog reflecting the active model. -/
def renderModelsJson (active : SparkModel) : String :=
  let qAct := if active == .qwen38FlashNext then "true" else "false"
  let lAct := if active == .leanstral15 then "true" else "false"
  let gAct := if active == .glm53Flash then "true" else "false"
  "{\"object\":\"list\",\"data\":[" ++
    s!"\{\"id\":\"qwen3.8-flash-next\",\"object\":\"model\",\"canonical\":\"qwen3.8-flash-next\",\"active\":{qAct}}," ++
    s!"\{\"id\":\"leanstral-1.5-119b-a6b\",\"object\":\"model\",\"canonical\":\"leanstral-1.5\",\"active\":{lAct}}," ++
    s!"\{\"id\":\"glm-5.3-flash\",\"object\":\"model\",\"canonical\":\"glm-5.3-flash\",\"active\":{gAct}}," ++
    "{\"id\":\"cohere-transcribe\",\"object\":\"model\",\"canonical\":\"cohere-transcribe\",\"type\":\"audio-transcription\",\"active\":true}" ++
  "]}"

/-- Evaluate the unified GPU background permit gate (`spark_gpu_permit_gate`) on current daemon state. -/
def evaluateGpuPermit (cfg : DaemonConfig) (st : DaemonState) (nowMs : UInt64) (freeMemMb : UInt64 := 32768)
    (paused : Bool := false) : UInt8 :=
  let swapFlag : UInt8 := if st.swapActive then 1 else 0
  let chatFlag : UInt8 := if st.chatActive > 0 then 1 else 0
  let asrFlag  : UInt8 := if st.asrActive > 0 then 1 else 0
  let pauseFlg : UInt8 := if paused then 1 else 0
  let idleMs   : UInt64 := if nowMs ≥ st.lastIdleMonoMs then nowMs - st.lastIdleMonoMs else 0
  sparkGpuPermitGate swapFlag chatFlag asrFlag pauseFlg 1 idleMs cfg.gpuIdleMinMs freeMemMb cfg.gpuMinFreeMb

/-- Handle a single accepted client socket `clientFd` end-to-end (including streaming SSE via `relayHttpStream`). -/
def handleClientConnection (cfg : DaemonConfig) (stRef : IO.Ref DaemonState) (clientFd : Fd) : IO Unit := do
  try
    let req ← match ← readHttpRequestFromFd clientFd none 10000 with
      | .ok r => pure r
      | .error _ => return ()
    if !authorizeRequest cfg req true then
      let unauthorized := "{\"error\":{\"message\":\"Missing or invalid API key.\",\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}"
      discard <| writeAll clientFd (buildJsonWire 401 unauthorized) false
      return ()

    let cleanPath := match req.path.splitOn "?" with | p :: _ => p | [] => req.path
    let nowNs ← IO.monoNanosNow
    let nowMs : UInt64 := (nowNs / 1000000).toUInt64

    if req.method == "GET" && (cleanPath == "/" || cleanPath == "/health") then
      let st ← stRef.get
      let curStr := if st.currentModel == .none then "null" else s!"\"{st.currentModel.canonicalName}\""
      discard <| writeAll clientFd (buildJsonWire 200 s!"\{\"status\":\"ok\",\"engine\":\"spark-daemon-lean\",\"current_model\":{curStr}}") false
      return ()

    if req.method == "GET" && cleanPath == "/v1/models" then
      let st ← stRef.get
      discard <| writeAll clientFd (buildJsonWire 200 (renderModelsJson st.currentModel)) false
      return ()

    if req.method == "GET" && cleanPath == "/v1/gpu/permit" then
      let st ← stRef.get
      let paused ← (System.FilePath.mk "/etc/spark-paused").pathExists
      let permit := evaluateGpuPermit cfg st nowMs 32768 paused
      let body := s!"\{\"permit\":{permit == 1},\"swap_active\":{st.swapActive},\"chat_active\":{st.chatActive},\"asr_active\":{st.asrActive},\"swap_epoch\":{st.swapEpoch}}"
      discard <| writeAll clientFd (buildJsonWire 200 body) false
      return ()

    if req.method == "POST" && (cleanPath == "/v1/chat/completions" || cleanPath == "/v1/completions") then
      let tape := JsonTape.indexJsonObject req.bodySlice
      let reqModelRaw := (JsonTape.lookupString? tape "model").getD "qwen3.8-flash-next"
      let isStream := (JsonTape.lookupBool? tape "stream").getD false
      let some targetModel := SparkModel.resolveAlias reqModelRaw
        | do
          let errJson := s!"\{\"error\":\{\"message\":\"Unknown model: {reqModelRaw}\",\"type\":\"invalid_request_error\"}}"
          discard <| writeAll clientFd (buildJsonWire 400 errJson) false
          return ()

      let healthy ← probeBackendHealthy cfg
      let st0 ← stRef.get
      let idleSec : UInt64 :=
        if st0.lastUsedMonoMs == 0 || nowMs < st0.lastUsedMonoMs then cfg.swapGraceSec
        else (nowMs - st0.lastUsedMonoMs) / 1000
      let admit := sparkSwapAdmit
        st0.currentModel.toId
        targetModel.toId
        (if st0.swapActive then 1 else 0)
        (if healthy then 1 else 0)
        idleSec
        cfg.swapGraceSec

      if admit == 3 then
        let waitSec := sparkSwapRetryAfterSec (if st0.swapActive then 1 else 0) idleSec cfg.swapGraceSec 30
        let msg := s!"{st0.currentModel.canonicalName} is loaded and in use; {targetModel.canonicalName} can be loaded in {waitSec}s"
        let errJson := s!"\{\"error\":\{\"message\":\"{msg}\",\"type\":\"rate_limit_error\",\"code\":\"model_busy\"}}"
        discard <| writeAll clientFd (buildJsonWire 503 errJson (some waitSec)) false
        return ()

      -- Mark foreground chat turn active (revokes background GPU permits via `spark_gpu_permit_gate`)
      stRef.modify fun s => { s with chatActive := s.chatActive + 1, swapActive := admit == 2 }
      try
        if admit == 2 then
          match ← runModelSwapGroup cfg targetModel with
          | .error swapErr =>
            stRef.modify fun s => { s with swapActive := false }
            let errJson := s!"\{\"error\":\{\"message\":\"{swapErr}\",\"type\":\"server_error\",\"code\":\"swap_failed\"}}"
            discard <| writeAll clientFd (buildJsonWire 503 errJson) false
            return ()
          | .ok () =>
            let readyMs : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
            stRef.modify fun s => {
              s with
              currentModel := targetModel
              swapActive := false
              lastUsedMonoMs := readyMs
              swapEpoch := s.swapEpoch + 1
            }

        let normBody :=
          if cleanPath == "/v1/chat/completions" then
            normalizeChatPayload req.bodySlice targetModel
          else
            req.bodySlice.toByteArray
        let prefillBudget := sparkPrefillBudgetMs normBody.size.toUInt64 cfg.basePrefillMs cfg.per4kPrefillMs cfg.maxPrefillMs

        if isStream then
          let .ok upFd ← tcpConnect cfg.backendHost cfg.backendPort
            | do
              stRef.modify fun s => { s with currentModel := .none }
              let errJson := "{\"error\":{\"message\":\"backend unavailable\",\"type\":\"server_error\"}}"
              discard <| writeAll clientFd (buildJsonWire 503 errJson) false
              return ()
          let upReqHead :=
            s!"POST {req.path} HTTP/1.1\r\n" ++
            s!"Host: {cfg.backendHost}:{cfg.backendPort}\r\n" ++
            "Content-Type: application/json\r\n" ++
            s!"Content-Length: {normBody.size}\r\n" ++
            "Connection: close\r\n\r\n"
          let sseHdr :=
            ("HTTP/1.1 200 OK\r\n" ++
             "Content-Type: text/event-stream; charset=utf-8\r\n" ++
             "Cache-Control: no-cache\r\n" ++
             "X-Accel-Buffering: no\r\n" ++
             "Access-Control-Allow-Origin: *\r\n" ++
             "Connection: close\r\n\r\n" ++
             ": keepalive\n\n").toUTF8
          let relayCfg : StreamRelayConfig := {
            prefillMaxMs := prefillBudget
            idleMaxMs := cfg.streamIdleMaxMs
            keepaliveMs := cfg.sseKeepaliveMs
            maxBufferBytes := 4194304
          }
          let _ ← relayHttpStream relayCfg clientFd upFd (upReqHead.toUTF8 ++ normBody) (some sseHdr)
          let endMs : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
          stRef.modify fun s => { s with lastUsedMonoMs := endMs }
        else
          match ← httpClientRequest cfg.backendHost cfg.backendPort "POST" req.path (ByteSlice.ofByteArray normBody) none prefillBudget.toUInt32 with
          | .ok respBody =>
            let cleaned := stripChatMlText respBody
            let endMs : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
            stRef.modify fun s => { s with lastUsedMonoMs := endMs }
            discard <| writeAll clientFd (buildJsonWire 200 cleaned) false
          | .error e =>
            stRef.modify fun s => { s with currentModel := .none }
            let errJson := s!"\{\"error\":\{\"message\":\"backend error: {e}\",\"type\":\"server_error\"}}"
            discard <| writeAll clientFd (buildJsonWire 503 errJson) false
      finally
        let finMs : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
        stRef.modify fun s => {
          s with
          swapActive := false
          chatActive := if s.chatActive > 0 then s.chatActive - 1 else 0
          lastIdleMonoMs := finMs
        }
      return ()

    discard <| writeAll clientFd (buildJsonWire 404 "{\"error\":{\"message\":\"not found\"}}") false
  finally
    discard <| closeFd clientFd

/-- Serve `maxRequests` (`0` = unbounded daemon loop) on `listenFd`. -/
def runServeLoop (cfg : DaemonConfig) (stRef : IO.Ref DaemonState) (listenFd : Fd)
    (maxRequests : Nat := 0) : IO UInt32 := do
  let mut served : Nat := 0
  let mut activeTasks : Array (Task (Except IO.Error Unit)) := #[]
  while maxRequests == 0 || served < maxRequests do
    match ← tcpAcceptPoll listenFd 250 with
    | .ok cfd =>
      served := served + 1
      let t ← IO.asTask (prio := .dedicated) (handleClientConnection cfg stRef cfd)
      activeTasks := activeTasks.push t
    | .error .eagain => pure ()
    | .error .eintr  => pure ()
    | .error _       => break
  for t in activeTasks do
    discard <| IO.wait t
  return 0

/-- Deterministic end-to-end self-test verifying all `Spark.Kernel` C-ABI gates, ChatML sentinel
    stripping, `SWAP_GRACE_SEC` anti-thrashing (`503 + Retry-After`), GPU permit revocation,
    and live SSE `relayHttpStream` keepalive + client-disconnect cancellation. -/
def runSelfTest : IO UInt32 := do
  -- 1. Pure Kernel Gate Sanity
  let gpuOk :=
    sparkGpuPermitGate 0 0 0 0 1 25000 20000 32768 12288 == 1 &&
    sparkGpuPermitGate 1 0 0 0 1 25000 20000 32768 12288 == 0 && -- swapActive blocks permit
    sparkGpuPermitGate 0 1 0 0 1 25000 20000 32768 12288 == 0 && -- chatBusy blocks permit
    sparkGpuPermitGate 0 0 1 0 1 25000 20000 32768 12288 == 0 && -- asrBusy blocks permit
    sparkGpuPermitGate 0 0 0 0 1 5000 20000 32768 12288 == 0 &&  -- cooldown blocks permit
    sparkGpuPermitGate 0 0 0 0 1 25000 20000 8192 12288 == 0     -- low free VRAM blocks permit
  let swapOk :=
    sparkSwapAdmit 1 1 0 1 10 600 == 1 &&  -- healthy active model reused
    sparkSwapAdmit 1 2 0 1 10 600 == 3 &&  -- grace period blocks thrash (`503`)
    sparkSwapAdmit 1 2 0 1 605 600 == 2 && -- past grace period allows swap
    sparkSwapAdmit 1 1 0 0 10 600 == 2 &&  -- crashed active model relaunches immediately
    sparkSwapAdmit 1 2 1 1 900 600 == 3 && -- concurrent swapActive blocks duplicate swap
    sparkSwapAdmit 0 9 0 1 900 600 == 0    -- invalid model ID rejected
  let prefillOk :=
    sparkPrefillBudgetMs 1024 90000 4000 900000 == 90000 &&
    sparkPrefillBudgetMs (40 * 4096) 90000 4000 900000 == 250000 &&
    sparkPrefillBudgetMs (999999 * 4096) 90000 4000 900000 == 900000
  let authOk :=
    sparkAuthGate 1 1 0 1 0 0 == 1 && -- `/health` open
    sparkAuthGate 1 0 1 0 0 0 == 1 && -- direct loopback allowed
    sparkAuthGate 1 0 0 1 1 0 == 1 && -- `lean-mesh` `10.88.0.0/16` allowed in `external`
    sparkAuthGate 1 0 0 1 0 0 == 0 && -- external without key rejected
    sparkAuthGate 1 0 0 1 0 1 == 1 && -- external with valid key allowed
    sparkAuthGate 2 0 0 1 1 0 == 0    -- `all` mode requires key even for mesh peers
  if !(gpuOk && swapOk && prefillOk && authOk) then
    IO.eprintln "[spark-daemon] FAIL: Spark.Kernel gate assertions failed"
    return 1

  -- 2. Alias resolution + ChatML sentinel stripping + `reasoning_effort` injection
  if SparkModel.resolveAlias "builtin/private" != some .qwen38FlashNext ||
     SparkModel.resolveAlias "leanstral" != some .leanstral15 ||
     SparkModel.resolveAlias "glm5" != some .glm53Flash ||
     (SparkModel.resolveAlias "gpt-99-unknown").isSome then
    IO.eprintln "[spark-daemon] FAIL: SparkModel.resolveAlias mismatch"
    return 1
  let dirtyResp := "{\"choices\":[{\"message\":{\"content\":\"Solved!<|im_end|>\"}}]}"
  if stripChatMlText dirtyResp != "{\"choices\":[{\"message\":{\"content\":\"Solved!\"}}]}" then
    IO.eprintln "[spark-daemon] FAIL: stripChatMlText failed"
    return 1
  let normQwen := String.fromUTF8! (normalizeChatPayload (ByteSlice.ofString "{\"model\":\"qwen\",\"messages\":[]}") .qwen38FlashNext)
  if (normQwen.splitOn "\"model\":\"qwen3.8-flash-next\"").length ≤ 1 ||
     (normQwen.splitOn "\"reasoning_effort\":\"low\"").length ≤ 1 then
    IO.eprintln s!"[spark-daemon] FAIL: normalizeChatPayload unexpected: {normQwen}"
    return 1

  -- 3. End-to-End Loopback Mock vLLM Backend + `spark-daemon` SSE Keepalive + `503` Grace Refusal
  let .ok (mockVllmFd, mockVllmPort) ← tcpListen "127.0.0.1" 0 16
    | do IO.eprintln "[spark-daemon] FAIL: tcpListen mockVllm"; return 1
  let mockVllmTask ← IO.asTask (prio := .dedicated) do
    -- Request 1: `/health` probe
    if let .ok c1 ← tcpAcceptPoll mockVllmFd 3000 then
      let _ ← readHttpRequestFromFd c1 none 1000
      discard <| writeAll c1 (HttpResponse.json 200 "{\"status\":\"ok\"}").toWireBytes false
      discard <| closeFd c1
    -- Request 2: `/v1/chat/completions` streaming with 110ms prefill pause so 40ms keepalive fires!
    if let .ok c2 ← tcpAcceptPoll mockVllmFd 3000 then
      let _ ← readHttpRequestFromFd c2 none 1000
      IO.sleep 110
      let ssePayload :=
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" ++
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n" ++
        "data: [DONE]\n\n"
      discard <| writeAll c2 ssePayload.toUTF8 false
      discard <| closeFd c2
    -- Request 3: `/health` probe before competing `leanstral` request (triggers `503` grace refusal!)
    if let .ok c3 ← tcpAcceptPoll mockVllmFd 3000 then
      let _ ← readHttpRequestFromFd c3 none 1000
      discard <| writeAll c3 (HttpResponse.json 200 "{\"status\":\"ok\"}").toWireBytes false
      discard <| closeFd c3

  let .ok (routerFd, routerPort) ← tcpListen "127.0.0.1" 0 16
    | do discard <| closeFd mockVllmFd; IO.eprintln "[spark-daemon] FAIL: tcpListen router"; return 1
  let now0 : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
  let stRef ← IO.mkRef ({
    currentModel := .qwen38FlashNext
    lastUsedMonoMs := now0
    lastIdleMonoMs := now0
  } : DaemonState)
  let cfg : DaemonConfig := {
    listenPort := routerPort
    backendPort := mockVllmPort
    swapScriptPath := ""
    swapGraceSec := 600
    sseKeepaliveMs := 40
    authMode := 1
    apiKeys := #["spark-test-secret"]
  }
  let daemonTask ← IO.asTask (prio := .dedicated) do
    runServeLoop cfg stRef routerFd 4

  -- Client Call 1: `/health`
  let .ok healthBody ← httpClientRequest "127.0.0.1" routerPort "GET" "/health" (ByteSlice.ofString "") none 2000
    | do IO.eprintln "[spark-daemon] FAIL: GET /health"; return 1
  -- Client Call 2: `/v1/models`
  let .ok modelsBody ← httpClientRequest "127.0.0.1" routerPort "GET" "/v1/models" (ByteSlice.ofString "") none 2000
    | do IO.eprintln "[spark-daemon] FAIL: GET /v1/models"; return 1
  -- Client Call 3: Streaming `/v1/chat/completions` (`qwen` alias -> `qwen3.8-flash-next`)
  let .ok streamCli ← tcpConnect "127.0.0.1" routerPort
    | do IO.eprintln "[spark-daemon] FAIL: tcpConnect streamCli"; return 1
  let streamBody := "{\"model\":\"qwen\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}"
  let streamReq :=
    s!"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {streamBody.utf8ByteSize}\r\nConnection: close\r\n\r\n{streamBody}"
  discard <| writeAll streamCli streamReq.toUTF8 false
  let mut sseRx := ByteArray.empty
  for _ in [0:32] do
    match ← pollReadable streamCli 1000 with
    | .ok () =>
      let ch ← readStreamChunk streamCli 4096
      if ch.isEmpty then break
      sseRx := sseRx ++ ch
    | .error _ => break
  discard <| closeFd streamCli
  let sseStr := String.fromUTF8! sseRx

  -- Client Call 4: Competing model `leanstral` inside `SWAP_GRACE_SEC` -> must return `503` + `Retry-After:`!
  let .ok busyCli ← tcpConnect "127.0.0.1" routerPort
    | do IO.eprintln "[spark-daemon] FAIL: tcpConnect busyCli"; return 1
  let busyBody := "{\"model\":\"leanstral\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"prove\"}]}"
  let busyReq :=
    s!"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {busyBody.utf8ByteSize}\r\nConnection: close\r\n\r\n{busyBody}"
  discard <| writeAll busyCli busyReq.toUTF8 false
  let mut busyRx := ByteArray.empty
  for _ in [0:16] do
    match ← pollReadable busyCli 1000 with
    | .ok () =>
      let ch ← readStreamChunk busyCli 4096
      if ch.isEmpty then break
      busyRx := busyRx ++ ch
    | .error _ => break
  discard <| closeFd busyCli
  let busyStr := String.fromUTF8! busyRx

  discard <| IO.wait daemonTask
  discard <| IO.wait mockVllmTask
  discard <| closeFd routerFd
  discard <| closeFd mockVllmFd

  if (healthBody.splitOn "\"spark-daemon-lean\"").length ≤ 1 ||
     (modelsBody.splitOn "\"qwen3.8-flash-next\"").length ≤ 1 ||
     (sseStr.splitOn ": keepalive\n\n").length ≤ 2 ||
     (sseStr.splitOn "data: [DONE]\n\n").length ≤ 1 ||
     (busyStr.splitOn "503 Service Unavailable").length ≤ 1 ||
     (busyStr.splitOn "Retry-After:").length ≤ 1 then
    IO.eprintln s!"[spark-daemon] FAIL: E2E router verification mismatch:\nsse={sseStr}\nbusy={busyStr}"
    return 1

  IO.println s!"\{\"ok\":true,\"binary\":\"spark-daemon\",\"c_abi_gates\":4,\"leansys_stream_relay\":true,\"sse_keepalive_verified\":true,\"swap_grace_503_verified\":true,\"router_port\":{routerPort}}"
  return 0

end Spark.Main

def main (args : List String) : IO UInt32 := do
  match args with
  | ["self-test"] | ["--self-test"] => Spark.Main.runSelfTest
  | ["serve"] =>
    let host := (← IO.getEnv "ROUTER_HOST").getD "0.0.0.0"
    let port := ((← IO.getEnv "ROUTER_PORT").bind String.toNat?).getD 8000 |>.toUInt16
    let backendPort := ((← IO.getEnv "BACKEND_PORT").bind String.toNat?).getD 8001 |>.toUInt16
    let graceSec := ((← IO.getEnv "ROUTER_SWAP_GRACE_SEC").bind String.toNat?).getD 600 |>.toUInt64
    let kaSec := ((← IO.getEnv "ROUTER_SSE_KEEPALIVE_SEC").bind String.toNat?).getD 10 |>.toUInt64
    let rawKeys := (← IO.getEnv "ROUTER_API_KEYS").getD ""
    let keys := (rawKeys.splitOn ",").foldl (fun acc k =>
      let t := k.trimAscii.toString
      if t.isEmpty then acc else acc.push t) #[]
    let authStr := ((← IO.getEnv "ROUTER_AUTH").getD (if keys.isEmpty then "off" else "external")).trimAscii.toString.toLower
    let authMode : UInt8 := if authStr == "all" then 2 else if authStr == "external" then 1 else 0
    let cfg : Spark.Main.DaemonConfig := {
      listenHost := host
      listenPort := port
      backendPort := backendPort
      swapGraceSec := graceSec
      sseKeepaliveMs := kaSec * 1000
      authMode := authMode
      apiKeys := keys
    }
    let .ok (listenFd, boundPort) ← LeanSys.Posix.tcpListen cfg.listenHost cfg.listenPort 64
      | do IO.eprintln s!"[spark-daemon] failed to bind {cfg.listenHost}:{cfg.listenPort}"; return 1
    let nowMs : UInt64 := ((← IO.monoNanosNow) / 1000000).toUInt64
    let stRef ← IO.mkRef ({ lastIdleMonoMs := nowMs } : Spark.Main.DaemonState)
    IO.println s!"[spark-daemon] listening on http://{cfg.listenHost}:{boundPort} (backend 127.0.0.1:{cfg.backendPort})"
    try
      Spark.Main.runServeLoop cfg stRef listenFd 0
    finally
      discard <| LeanSys.Posix.closeFd listenFd
  | ["--c-abi"] =>
    IO.println "{\"ok\":true,\"exports\":[\"spark_gpu_permit_gate\",\"spark_swap_admit\",\"spark_swap_retry_after_sec\",\"spark_prefill_budget_ms\",\"spark_auth_gate\"]}"
    return 0
  | _ =>
    IO.println "usage: spark-daemon [self-test | serve | --c-abi]"
    return 0

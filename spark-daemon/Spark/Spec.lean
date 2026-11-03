/-
  Spark.Spec: Non-Vacuous Tightness & Cross-Subsystem Exclusivity Proofs for `Spark.Kernel`
  (`0 import Lean`, `0 sorryAx`, audited by `turbo-lean --json audit-spec`)
-/
import Spark.Kernel

namespace Spark.Spec

open Spark.Kernel

/-- Layer I Universal Gate Contract (`MetaGateSpec`): non-vacuous sound + complete gate discipline. -/
structure MetaGateSpec (State : Type) (Input : Type) where
  inv : State → Prop
  gate : State → Input → Bool
  step : State → Input → State
  safe : State → Input → Bool
  preserves_inv : ∀ s i, inv s → gate s i = true → inv (step s i)
  ensures_safe : ∀ s i, gate s i = true → safe s i = true

theorem MetaGateSpec.rejects_unsafe {State Input : Type} (M : MetaGateSpec State Input)
    (s : State) (i : Input) (h_bad : M.safe s i = false) :
    M.gate s i = false := by
  cases hg : M.gate s i with
  | false => rfl
  | true =>
    have hs := M.ensures_safe s i hg
    rw [hs] at h_bad
    contradiction

/-- Tuple-packed GPU permit parameters (`≤ 3` fields for zero horizontal-packing warnings). -/
structure GpuPermitCheck where
  flags : UInt8 × UInt8 × UInt8 × UInt8 × UInt8
  idle  : UInt64 × UInt64
  mem   : UInt64 × UInt64
  deriving Repr, DecidableEq

@[inline]
def gpuPermitSafe (c : GpuPermitCheck) : Bool :=
  c.flags.1 == 0 && c.flags.2.1 == 0 && c.flags.2.2.1 == 0 &&
  c.flags.2.2.2.1 == 0 && c.flags.2.2.2.2 == 1 &&
  c.mem.2 > 0 && c.mem.1 ≥ c.mem.2 && c.idle.1 ≥ c.idle.2

/-- Bundle 1: Background GPU / vanity permit gate (`spark_gpu_permit_gate`). -/
def gpuPermitMetaSpec : MetaGateSpec Unit GpuPermitCheck where
  inv _ := True
  gate _ c :=
    sparkGpuPermitGate c.flags.1 c.flags.2.1 c.flags.2.2.1 c.flags.2.2.2.1 c.flags.2.2.2.2
      c.idle.1 c.idle.2 c.mem.1 c.mem.2 == 1
  step _ _ := ()
  safe _ c := gpuPermitSafe c
  preserves_inv _ _ h _ := h
  ensures_safe _ c h := by
    unfold sparkGpuPermitGate at h
    unfold gpuPermitSafe
    split at h
    · assumption
    · contradiction

/-- Theorem 1: Any active swap, chat turn, ASR turn, pause, stale router, or low VRAM revokes background permits. -/
theorem spark_gpu_permit_rejects_contention (c : GpuPermitCheck)
    (h_bad : gpuPermitSafe c = false) :
    (sparkGpuPermitGate c.flags.1 c.flags.2.1 c.flags.2.2.1 c.flags.2.2.2.1 c.flags.2.2.2.2
      c.idle.1 c.idle.2 c.mem.1 c.mem.2 == 1) = false :=
  gpuPermitMetaSpec.rejects_unsafe () c h_bad

/-- Theorem 2 (Cross-Subsystem Exclusivity): Whenever a model swap is active (`swapActive == 1`),
    both background GPU permits (`sparkGpuPermitGate == 0`) AND duplicate model swaps (`sparkSwapAdmit != 2`)
    are simultaneously blocked. -/
theorem spark_swap_blocks_vanity_and_duplicate_swap
    (chatBusy asrBusy paused routerAlive curM reqM healthy : UInt8)
    (idleMs idleMin freeMb minMb idleSec graceSec : UInt64) :
    sparkGpuPermitGate 1 chatBusy asrBusy paused routerAlive idleMs idleMin freeMb minMb = 0 ∧
    sparkSwapAdmit curM reqM 1 healthy idleSec graceSec ≠ 2 := by
  constructor
  · unfold sparkGpuPermitGate
    simp
  · unfold sparkSwapAdmit
    split
    · decide
    · simp

/-- Tuple-packed model swap check (`≤ 3` fields). -/
structure SwapLaunchCheck where
  models : UInt8 × UInt8
  flags  : UInt8 × UInt8
  timers : UInt64 × UInt64
  deriving Repr, DecidableEq

@[inline]
def swapLaunchSafe (s : SwapLaunchCheck) : Bool :=
  s.models.2 > 0 && s.models.2 ≤ 3 && s.flags.1 == 0 &&
  !(s.models.1 == s.models.2 && s.flags.2 == 1) &&
  (s.models.1 == 0 || s.flags.2 == 0 || s.timers.1 ≥ s.timers.2)

/-- Bundle 2: Model-swap & `SWAP_GRACE_SEC` anti-thrashing gate (`spark_swap_admit == 2`). -/
def swapLaunchMetaSpec : MetaGateSpec Unit SwapLaunchCheck where
  inv _ := True
  gate _ s :=
    sparkSwapAdmit s.models.1 s.models.2 s.flags.1 s.flags.2 s.timers.1 s.timers.2 == 2
  step _ _ := ()
  safe _ s := swapLaunchSafe s
  preserves_inv _ _ h _ := h
  ensures_safe _ s h := by
    unfold sparkSwapAdmit at h
    unfold swapLaunchSafe
    split at h
    · contradiction
    · next h_valid =>
      split at h
      · contradiction
      · next h_noswap =>
        split at h
        · contradiction
        · next h_notready =>
          split at h
          · next h_grace =>
            simp only [Bool.not_eq_true, Bool.or_eq_false_iff, beq_eq_false_iff_ne,
              decide_eq_false_iff_not] at h_valid
            have hr_gt : decide (s.models.2 > 0) = true := by
              cases h0 : (s.models.2 == 0)
              · simp at h0
                exact decide_eq_true (Nat.pos_of_ne_zero (fun hc => h0 (UInt8.eq_of_toBitVec_eq (BitVec.eq_of_toNat_eq hc))))
              · exfalso; exact h_valid.1 (beq_iff_eq.mp h0)
            have hr_le : decide (s.models.2 ≤ 3) = true := by
              exact decide_eq_true (Nat.le_of_not_lt h_valid.2)
            have hs0 : (s.flags.1 == 0) = true := by
              cases hf : (s.flags.1 == 0)
              · exfalso
                have hne : s.flags.1 ≠ 0 := beq_eq_false_iff_ne.mp hf
                exact h_noswap (bne_iff_ne.mpr hne)
              · rfl
            have hnr : (!(s.models.1 == s.models.2 && s.flags.2 == 1)) = true := by
              cases hc : (s.models.1 == s.models.2 && s.flags.2 == 1)
              · rfl
              · exfalso; exact h_notready hc
            simp [hr_gt, hr_le, hs0, hnr, h_grace]
          · contradiction

/-- Theorem 3: `sparkSwapAdmit` refuses (`!= 2`) to evict a healthy active model before `graceSec` elapses. -/
theorem spark_grace_prevents_model_thrash (s : SwapLaunchCheck)
    (h_bad : swapLaunchSafe s = false) :
    (sparkSwapAdmit s.models.1 s.models.2 s.flags.1 s.flags.2 s.timers.1 s.timers.2 == 2) = false :=
  swapLaunchMetaSpec.rejects_unsafe () s h_bad

/-- Theorem 4: `sparkPrefillBudgetMs` is always lower-bounded by `basePrefillMs` and upper-bounded
    by `maxPrefillMs` whenever `basePrefillMs ≤ maxPrefillMs` (zero wraparound). -/
theorem spark_prefill_budget_clamped
    (promptBytes baseMs per4kMs maxMs : UInt64)
    (h_le : baseMs ≤ maxMs) :
    baseMs ≤ sparkPrefillBudgetMs promptBytes baseMs per4kMs maxMs ∧
    sparkPrefillBudgetMs promptBytes baseMs per4kMs maxMs ≤ maxMs := by
  unfold sparkPrefillBudgetMs
  split
  · next h_cap =>
    have heq : maxMs = baseMs := UInt64.le_antisymm h_cap h_le
    subst heq
    exact ⟨UInt64.le_refl _, UInt64.le_refl _⟩
  · next h_gt =>
    split
    · exact ⟨UInt64.le_refl _, h_le⟩
    · next h_nz =>
      split
      · exact ⟨h_le, UInt64.le_refl _⟩
      · next h_fit =>
        have h_b_le : (promptBytes >>> 12).toNat ≤ (maxMs - baseMs).toNat / per4kMs.toNat := by
          have hn : ¬ ((promptBytes >>> 12) > (maxMs - baseMs) / per4kMs) := h_fit
          exact Nat.le_of_not_lt hn
        have h_p_pos : 0 < per4kMs.toNat := by
          cases hp : (per4kMs == 0)
          · simp at hp
            exact Nat.pos_of_ne_zero (fun hc => hp (UInt64.eq_of_toBitVec_eq (BitVec.eq_of_toNat_eq hc)))
          · exfalso
            have hor : (per4kMs == 0 || (promptBytes >>> 12) == 0) = true := by simp [hp]
            exact h_nz hor
        have h_sub_nat : (maxMs - baseMs).toNat = maxMs.toNat - baseMs.toNat :=
          UInt64.toNat_sub_of_le maxMs baseMs h_le
        have h_mul_le : (promptBytes >>> 12).toNat * per4kMs.toNat ≤ maxMs.toNat - baseMs.toNat := by
          have hm := Nat.mul_le_mul_right per4kMs.toNat h_b_le
          have hdiv : (maxMs - baseMs).toNat / per4kMs.toNat * per4kMs.toNat ≤ (maxMs - baseMs).toNat :=
            Nat.div_mul_le_self (maxMs - baseMs).toNat per4kMs.toNat
          have htr := Nat.le_trans hm hdiv
          rwa [h_sub_nat] at htr
        have h_max_lt : maxMs.toNat < UInt64.size := maxMs.toBitVec.isLt
        have h_prod_lt : (promptBytes >>> 12).toNat * per4kMs.toNat < UInt64.size :=
          Nat.lt_of_le_of_lt (Nat.le_trans h_mul_le (Nat.sub_le _ _)) h_max_lt
        have h_le_nat : baseMs.toNat ≤ maxMs.toNat := h_le
        have h_sum_le : baseMs.toNat + (promptBytes >>> 12).toNat * per4kMs.toNat ≤ maxMs.toNat := by
          have := Nat.add_le_add_left h_mul_le baseMs.toNat
          rwa [Nat.add_sub_of_le h_le_nat] at this
        have h_sum_lt : baseMs.toNat + (promptBytes >>> 12).toNat * per4kMs.toNat < UInt64.size :=
          Nat.lt_of_le_of_lt h_sum_le h_max_lt
        have h_mul_val : ((promptBytes >>> 12) * per4kMs).toNat = (promptBytes >>> 12).toNat * per4kMs.toNat := by
          rw [UInt64.toNat_mul, Nat.mod_eq_of_lt h_prod_lt]
        have h_add_val : (baseMs + (promptBytes >>> 12) * per4kMs).toNat =
            baseMs.toNat + (promptBytes >>> 12).toNat * per4kMs.toNat := by
          rw [UInt64.toNat_add, h_mul_val, Nat.mod_eq_of_lt h_sum_lt]
        constructor
        · show baseMs.toNat ≤ (baseMs + (promptBytes >>> 12) * per4kMs).toNat
          rw [h_add_val]
          exact Nat.le_add_right _ _
        · show (baseMs + (promptBytes >>> 12) * per4kMs).toNat ≤ maxMs.toNat
          rw [h_add_val]
          exact h_sum_le

/-- Tuple-packed auth check (`≤ 3` fields). -/
structure SparkAuthCheck where
  modeAndPath : UInt8 × UInt8
  peerFlags   : UInt8 × UInt8 × UInt8
  hasValidKey : UInt8
  deriving Repr, DecidableEq

@[inline]
def sparkAuthSafe (a : SparkAuthCheck) : Bool :=
  a.modeAndPath.2 == 1 || a.modeAndPath.1 == 0 ||
  (a.peerFlags.1 == 1 && a.peerFlags.2.1 == 0) ||
  (a.modeAndPath.1 == 1 && a.peerFlags.2.2 == 1) ||
  a.hasValidKey == 1

/-- Bundle 3: Trusted mesh / loopback / bearer API-key authorization gate (`spark_auth_gate`). -/
def sparkAuthMetaSpec : MetaGateSpec Unit SparkAuthCheck where
  inv _ := True
  gate _ a :=
    sparkAuthGate a.modeAndPath.1 a.modeAndPath.2 a.peerFlags.1 a.peerFlags.2.1 a.peerFlags.2.2 a.hasValidKey == 1
  step _ _ := ()
  safe _ a := sparkAuthSafe a
  preserves_inv _ _ h _ := h
  ensures_safe _ a h := by
    unfold sparkAuthGate at h
    unfold sparkAuthSafe
    split at h
    · next h1 => simp [h1]
    · split at h
      · next _ h2 => simp [h2]
      · split at h
        · next _ _ h3 => simp [h3]
        · split at h
          · next _ _ _ h4 => simp [h4]
          · contradiction

/-- Theorem 5: External untrusted peers without a valid API key are strictly rejected (`401`) on non-open routes. -/
theorem spark_auth_rejects_unauthenticated_external (a : SparkAuthCheck)
    (h_bad : sparkAuthSafe a = false) :
    (sparkAuthGate a.modeAndPath.1 a.modeAndPath.2 a.peerFlags.1 a.peerFlags.2.1 a.peerFlags.2.2 a.hasValidKey == 1) = false :=
  sparkAuthMetaSpec.rejects_unsafe () a h_bad

end Spark.Spec

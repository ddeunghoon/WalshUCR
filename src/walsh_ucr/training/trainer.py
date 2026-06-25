from typing import Callable, Any, Optional
from dataclasses import dataclass
import jax.numpy as jnp
import optax
import jax
jax.config.update("jax_enable_x64", True)

import time
import logging

logger = logging.getLogger(__name__)

class JAX_Debug_Trainer:
    def __init__(
        self,
        train_cost_fn: Callable[..., Any],
        theta_init: Any,
        optimizer_name: str = "adam",
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.0,
        memory_size: int = 10,
        eval_interval: int = 20,
        jit_backend: Optional[str] = "gpu",
        eval_cost_fn: Optional[Callable[..., Any]] = None,
        n_outcome: Optional[int] = None,
        a_priori_probs: Optional[jnp.ndarray] = None,
        **kwargs
    ):
        self.train_cost_fn = train_cost_fn
        self.theta = theta_init
        self.optimizer_name = str(optimizer_name).lower()
        self.learning_rate = learning_rate
        self.weight_decay = float(weight_decay)
        self.memory_size = memory_size
        self.eval_interval = int(eval_interval)
        self.jit_backend = jit_backend
        
        # Optional/Legacy args
        self.eval_cost_fn = eval_cost_fn
        self.n_outcome = n_outcome
        self.a_priori_probs = a_priori_probs

        if self.optimizer_name == "adam":
            lr = self.learning_rate if self.learning_rate is not None else 1e-2
            self.optimizer = optax.adam(learning_rate=lr)
        elif self.optimizer_name == "adamw":
            lr = self.learning_rate if self.learning_rate is not None else 1e-2
            self.optimizer = optax.adamw(learning_rate=lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "l-bfgs":
            lr = self.learning_rate if self.learning_rate is not None else 1.0
            self.optimizer = optax.lbfgs(learning_rate=lr, memory_size=self.memory_size)
        elif self.optimizer_name in ("sgd", "gd"):
            lr = self.learning_rate if self.learning_rate is not None else 1.0
            self.optimizer = optax.sgd(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.opt_state = self.optimizer.init(self.theta)
        
        self._loss_and_grad = jax.value_and_grad(self.train_cost_fn, argnums=0)
        
        self._loss_and_grad_compiled = self._maybe_jit(self._loss_and_grad)
        self._eval_compiled = self._maybe_jit(self.eval_cost_fn) if self.eval_cost_fn else None

    def _maybe_jit(self, fn):
        try:
            return jax.jit(fn, backend=self.jit_backend) if self.jit_backend else jax.jit(fn)
        except Exception as e:
            logger.warning(
                "[JIT disabled] %s (%s)",
                fn.__name__,
                type(e).__name__,
            )
            return fn

    @staticmethod
    def _extract_loss(eval_out):
        return eval_out[0] if isinstance(eval_out, tuple) else eval_out

    def run_optimization(
        self,
        steps: int,
        train_args: tuple,
        eval_args: Optional[tuple] = None,
        threshold: float = 1e-10,
        verbose: int = 1,
        eval_interval: Optional[int] = None,
        **kwargs
    ) -> "TrainResult":
        if eval_interval is not None:
            self.eval_interval = int(eval_interval)

        t0 = time.perf_counter()
        
        loss_log = []
        step_log = []

        L0, g0 = self._loss_and_grad_compiled(self.theta, *train_args)
        last_loss = float(L0)
        
        loss_log.append(last_loss)
        step_log.append(0)

        if verbose >= 1:
            print(f"[warmup] loss={last_loss:.6f}  first_call={(time.perf_counter() - t0):.5f} s")

        stopped_early = False
        nan_found = False
        last_step = 0

        for step in range(1, steps + 1):
            last_step = step
            
            L, grads = self._loss_and_grad_compiled(self.theta, *train_args)
            
            if self.optimizer_name == "l-bfgs":
                def value_fn(p):
                    return self.train_cost_fn(p, *train_args)
                
                updates, self.opt_state = self.optimizer.update(
                    grads, 
                    self.opt_state, 
                    self.theta, 
                    value=L, 
                    grad=grads, 
                    value_fn=value_fn
                )
            else:
                updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.theta)

            self.theta = optax.apply_updates(self.theta, updates)
            loss_val = float(L)

            if (step % self.eval_interval == 0):
                if verbose >= 1:
                    print(f"[{step}] loss={loss_val:.10f}  elapsed={(time.perf_counter()-t0):.5f} s")
                
                loss_log.append(loss_val)
                step_log.append(step)
                
                if jnp.isnan(loss_val):
                     if verbose >= 1:
                         print(f"Loss is NaN at step {step}")
                     nan_found = True
                     break

                diff = abs(last_loss - loss_val)
                if threshold is not None and diff <= float(threshold):
                    if verbose >= 1:
                        print(f"[early stop] step={step} loss={loss_val:.10f} diff={diff:.10f} <= {float(threshold):.10f}")
                    last_loss = loss_val
                    stopped_early = True
                    break
                
                last_loss = loss_val
        
        if verbose >= 1:
            print(f"Optimization finished: {time.perf_counter() - t0:.4f}s")
        
        return TrainResult(
            theta=self.theta,
            steps_run=last_step,
            last_eval_loss=last_loss,
            stopped_early=stopped_early,
            nan_found=nan_found,
            step_log=jnp.array(step_log),
            loss_log=jnp.array(loss_log)
        )

@jax.tree_util.register_pytree_node_class
@dataclass
class TrainResult:
    theta: Any
    steps_run: int
    last_eval_loss: float
    stopped_early: bool
    nan_found: bool
    step_log: Any   # jnp.ndarray[int32]
    loss_log: Any   # jnp.ndarray[float]

    # pytree support (so you can jit/pmapped if needed)
    def tree_flatten(self):
        children = (self.theta, self.step_log, self.loss_log)
        aux = (self.steps_run, self.last_eval_loss, self.stopped_early, self.nan_found)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        theta, step_log, loss_log = children
        steps_run, last_eval_loss, stopped_early, nan_found = aux
        return cls(theta, steps_run, last_eval_loss, stopped_early, nan_found, step_log, loss_log)


class JAX_Full_Trainer:
    """
    - 완전 JAX 루프: while_loop + jit
    - L-BFGS: optax.lbfgs + value_and_grad_from_state + value_fn (line search 효율)
    - 로그는 디바이스 배열에 기록 후, 종료 후 한 번에 호스트로 가져와 저장/출력
    """

    OPT_ADAM = 0
    OPT_LBFGS = 1

    def __init__(
        self,
        train_cost_fn: Callable[..., jnp.ndarray],   # train_cost_fn(theta, *train_args) -> scalar
        theta_init: Any,
        optimizer_name: str = "l-bfgs",
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.0,
        memory_size: int = 10,
        eval_interval: int = 50,
        enable_x64: bool = True,
        jit_backend: Optional[str] = "gpu",
        eval_cost_fn: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        if enable_x64:
            jax.config.update("jax_enable_x64", True)

        self.train_cost_fn = train_cost_fn
        self.eval_cost_fn = eval_cost_fn
        self.theta = theta_init
        self.eval_interval = int(eval_interval)
        self.jit_backend = jit_backend
        self.memory_size = memory_size
        self.ema_decay = kwargs.get("ema_decay", 0.99)
        self.weight_decay = float(weight_decay)

        name = optimizer_name.lower()
        if name in ("adam",):
            self._opt_kind = self.OPT_ADAM
            lr = learning_rate if learning_rate is not None else 1e-2
            self.optimizer = optax.adam(learning_rate=lr)
        elif name in ("adamw",):
            self._opt_kind = self.OPT_ADAM
            lr = learning_rate if learning_rate is not None else 1e-2
            self.optimizer = optax.adamw(learning_rate=lr, weight_decay=self.weight_decay)
        elif name in ("l-bfgs", "lbfgs"):
            self._opt_kind = self.OPT_LBFGS
            lr = learning_rate if learning_rate is not None else 1.0
            self.optimizer = optax.lbfgs(learning_rate=lr, memory_size=self.memory_size)
        elif name in ("sgd", "gd"):
            self._opt_kind = self.OPT_ADAM
            lr = learning_rate if learning_rate is not None else 1.0
            self.optimizer = optax.sgd(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer_name: {optimizer_name}")

        self.opt_state = self.optimizer.init(self.theta)

        # ---- Dynamic Loss Setup ----
        self.alt_cost_fn = kwargs.get("alt_cost_fn", None)

        # ---- value_fn: train_args는 키워드로 전달 (클로저/파이썬 루프 생성 최소화) ----
        # If alt_cost_fn is absent, avoid lax.cond entirely so JAX never traces a None callable branch.
        if self.alt_cost_fn is None:
            def _value_fn(params, use_alt_loss_flag, train_args):
                _ = use_alt_loss_flag  # keep signature compatibility
                return self.train_cost_fn(params, *train_args)
        else:
            def _value_fn(params, use_alt_loss_flag, train_args):
                def normal_loss(_):
                    return self.train_cost_fn(params, *train_args)

                def alt_loss(_):
                    return self.alt_cost_fn(params, *train_args)

                # flag가 True면 alt_loss, False면 normal_loss 실행
                return jax.lax.cond(use_alt_loss_flag, alt_loss, normal_loss, operand=None)

        # value_fn 자체 jit
        self._value_fn_jit = self._jit(_value_fn)

        # ---- eval_fn setup ----
        if self.eval_cost_fn is not None:
            def _eval_fn(params, *, eval_args):
                out = self.eval_cost_fn(params, *eval_args)
                if isinstance(out, tuple):
                    return out[0]
                return out
            self._eval_fn_jit = self._jit(_eval_fn)
        else:
            self._eval_fn_jit = None

        # adam용: value_and_grad (argnums=0)
        self._v_and_g_adam = self._jit(jax.value_and_grad(_value_fn, argnums=0))

        # lbfgs용: state에 저장된 value/grad 재사용 가능
        self._v_and_g_lbfgs = optax.value_and_grad_from_state(self._value_fn_jit)

        # ---- solver 2종(optimizer별) 미리 jit ----
        self._solve_adam = self._build_solve_adam()
        self._solve_lbfgs = self._build_solve_lbfgs()

    # ---------------------------
    # Public API
    # ---------------------------
    def _jit(self, fn, **jit_kwargs):
        if self.jit_backend:
            jit_kwargs.setdefault("backend", self.jit_backend)
        return jax.jit(fn, **jit_kwargs)

    def run_optimization(
        self,
        steps: int,
        train_args: tuple,
        eval_args: Optional[tuple] = None,
        threshold: float = 1e-10,
        verbose: int = 1,
        eval_interval: Optional[int] = None,
        early_stop: bool = True,
        return_numpy_logs: bool = False,
        **kwargs
    ) -> TrainResult:
        steps = int(steps)
        ei = int(self.eval_interval if eval_interval is None else eval_interval)
        
        # Eval args default
        eval_args_pass = eval_args if eval_args is not None else train_args

        if self._opt_kind == self.OPT_ADAM:
            train_tolerance = kwargs.get("train_tolerance", 1e-4) # default tolerance
            switch_step = int(kwargs.get("switch_step", -1))
            if self.alt_cost_fn is None:
                # Dynamic switching is disabled when alt_cost_fn is not provided.
                switch_step = -1
            result = self._solve_adam(
                self.theta,
                self.opt_state,
                train_args,
                eval_args_pass,
                max_steps=steps,
                threshold=float(threshold),
                eval_interval=ei,
                early_stop=bool(early_stop),
                train_tolerance=float(train_tolerance),
                switch_step=switch_step,
            )
        else:
            result = self._solve_lbfgs(
                self.theta,
                self.opt_state,
                train_args,
                eval_args_pass,
                max_steps=steps,
                threshold=float(threshold),
                eval_interval=ei,
                early_stop=bool(early_stop),
            )

        # 동기화는 여기서 한 번만
        jax.block_until_ready(result["theta"])

        # 내부 상태 갱신
        self.theta = result["theta"]
        self.opt_state = result["_opt_state"]

        # 호스트로 로그 변환 옵션
        if return_numpy_logs:
            import numpy as np
            step_log = np.asarray(result["step_log"])
            loss_log = np.asarray(result["loss_log"])
        else:
            step_log = result["step_log"]
            loss_log = result["loss_log"]

        return TrainResult(
            theta=result["theta"],
            steps_run=int(result["steps_run"]),
            last_eval_loss=float(result["last_eval_loss"]),
            stopped_early=bool(result["stopped_early"]),
            nan_found=bool(result["nan_found"]),
            step_log=step_log,
            loss_log=loss_log,
        )

    def save_logs_npz(self, path: str, step_log, loss_log) -> None:
        """루프 종료 후 호스트에서 저장(권장)."""
        import numpy as np
        np.savez(path, step_log=np.asarray(step_log), loss_log=np.asarray(loss_log))

    # ---------------------------
    # Internal: JAX solvers
    # ---------------------------
    def _build_solve_adam(self):
        # NOTE: opt_state를 같이 반환하려면 result에 붙여서 반환(파이썬 객체는 반환 불가 -> pytree로)
        
        # --------------------------------------------------------------------------------
        # Case 1: Dynamic Loss Switching Enabled (alt_cost_fn exists)
        # --------------------------------------------------------------------------------
        if self.alt_cost_fn is not None:
            def solve(theta0, opt_state0, train_args, eval_args, *, max_steps: int, threshold: float, eval_interval: int, early_stop: bool, train_tolerance: float, switch_step: int):
                n_eval = max_steps // eval_interval + 1
                
                # 초기 플래그(False)
                # If switch_step > 0 and start step 0 >= switch_step (unlikely for 0 unless switch_step=0), set True
                use_alt_loss_0 = jnp.bool_(False)
                if switch_step > 0:
                     use_alt_loss_0 = (jnp.int32(0) >= switch_step)
                
                # v0, g0 계산
                v0, g0 = self._v_and_g_adam(theta0, use_alt_loss_0, train_args)

                if self._eval_fn_jit is not None:
                    v_eval_0 = self._eval_fn_jit(theta0, eval_args=eval_args)
                else:
                    v_eval_0 = v0

                loss_log = jnp.full((n_eval,), jnp.nan, dtype=v0.dtype).at[0].set(v_eval_0)
                step_log = jnp.full((n_eval,), -1, dtype=jnp.int32).at[0].set(jnp.int32(0))

                # carry:
                # theta, opt_state, step, last_eval_loss, eval_idx, loss_log, step_log, stopped, nan_found, 
                # use_alt_loss, ema_train_loss, last_eval_ema
                
                # Init EMA with v0
                ema_0 = v0
                last_eval_ema_0 = v0 # EMA at last eval step for comparison
                
                carry = (
                    theta0,
                    opt_state0,
                    jnp.int32(0),
                    v0,              # last_eval_loss (used for early stopping)
                    jnp.int32(1),
                    loss_log,
                    step_log,
                    jnp.bool_(False),
                    jnp.bool_(jnp.isnan(v0)),
                    use_alt_loss_0,
                    ema_0,
                    last_eval_ema_0
                )

                def cond(c):
                    _, _, step, _, _, _, _, stopped, nan_found, _, _, _ = c
                    return (step < max_steps) & (~stopped) & (~nan_found)

                def body(c):
                    theta, opt_state, step, last_eval_loss, eval_idx, loss_log, step_log, stopped, nan_found, use_alt_loss, ema_loss, last_eval_ema = c

                    # Step-based switching override
                    # If switch_step > 0, we forcibly adhere to step schedule.
                    # Otherwise, we respect the current flag (which is updated by tolerance logic).
                    use_alt_loss = jnp.where(switch_step > 0, step >= switch_step, use_alt_loss)

                    # 1. Gradient Step
                    v, g = self._v_and_g_adam(theta, use_alt_loss, train_args)
                    updates, opt_state2 = self.optimizer.update(g, opt_state, theta)
                    theta2 = optax.apply_updates(theta, updates)

                    # 2. Update EMA (Exponential Moving Average)
                    # New EMA = decay * old_ema + (1-decay) * current_loss
                    ema_loss2 = self.ema_decay * ema_loss + (1.0 - self.ema_decay) * v
                    
                    step2 = step + jnp.int32(1)
                    nan2 = nan_found | jnp.isnan(v)
                    do_eval = (step2 % jnp.int32(eval_interval)) == 0

                    def on_eval(_):
                        # Eval Step
                        if self._eval_fn_jit is not None:
                            v_post = self._eval_fn_jit(theta2, eval_args=eval_args)
                        else:
                            v_post = self._value_fn_jit(theta2, use_alt_loss, train_args)

                        loss_log2 = loss_log.at[eval_idx].set(v_post)
                        step_log2 = step_log.at[eval_idx].set(step2)
                        eval_idx2 = eval_idx + jnp.int32(1)

                        # Early Stop Check
                        diff_eval = jnp.abs(last_eval_loss - v_post)
                        stop_now = jnp.bool_(early_stop) & (diff_eval <= jnp.asarray(threshold, v_post.dtype))
                        nan_now = jnp.isnan(v_post)
                        
                        # Loss Switching Logic (EMA diff check)
                        # Compare current EMA with EMA at last eval step
                        # Because we only want to switch if the trend flattens *over an interval*
                        diff_ema = jnp.abs(ema_loss2 - last_eval_ema)
                        


                        # If diff < tolerance, switch to alt loss
                        # Once switched, stay switched (| use_alt_loss)
                        switch_cond = diff_ema < train_tolerance
                        # If diff < tolerance, switch to alt loss
                        # Once switched, stay switched (| use_alt_loss)
                        switch_cond = diff_ema < train_tolerance
                        new_use_alt_loss_tol = use_alt_loss | switch_cond
                        
                        # Apply swtich_step priority
                        new_use_alt_loss = jnp.where(switch_step > 0, step2 >= switch_step, new_use_alt_loss_tol)
                        
                        # Update reference EMA for next interval
                        last_eval_ema2 = ema_loss2
                        
                        return v_post, eval_idx2, loss_log2, step_log2, stop_now, nan_now, new_use_alt_loss, last_eval_ema2

                    def on_no_eval(_):
                        # Keep state, but update switch if step-based
                        new_use_alt_loss = jnp.where(switch_step > 0, step2 >= switch_step, use_alt_loss)
                        return last_eval_loss, eval_idx, loss_log, step_log, jnp.bool_(False), jnp.bool_(False), new_use_alt_loss, last_eval_ema

                    (last_eval_loss2, eval_idx2, loss_log2, step_log2, stop_now, nan_now, 
                     use_alt_loss2, last_eval_ema2) = jax.lax.cond(
                        do_eval, on_eval, on_no_eval, operand=None
                    )

                    stopped2 = stopped | stop_now
                    nan2 = nan2 | nan_now
                    
                    return (theta2, opt_state2, step2, last_eval_loss2, eval_idx2, loss_log2, step_log2, stopped2, nan2, use_alt_loss2, ema_loss2, last_eval_ema2)

                theta_f, opt_state_f, step_f, last_eval_loss_f, eval_idx_f, loss_log_f, step_log_f, stopped_f, nan_f, _, _, _ = \
                    jax.lax.while_loop(cond, body, carry)

                return {
                    "theta": theta_f,
                    "_opt_state": opt_state_f,
                    "steps_run": step_f,
                    "last_eval_loss": last_eval_loss_f,
                    "loss_log": loss_log_f,
                    "step_log": step_log_f,
                    "stopped_early": stopped_f,
                    "nan_found": nan_f,
                }
            
            return self._jit(solve, static_argnames=("max_steps", "eval_interval", "early_stop", "switch_step"))

        # --------------------------------------------------------------------------------
        # Case 2: Standard Training (Disabled Overhead)
        # --------------------------------------------------------------------------------
        else:
            def solve(theta0, opt_state0, train_args, eval_args, *, max_steps: int, threshold: float, eval_interval: int, early_stop: bool, train_tolerance: float, switch_step: int):
                # train_tolerance is ignored here but kept for signature compatibility if needed
                # switch_step is ignored when alt_cost_fn is not set.
                n_eval = max_steps // eval_interval + 1
                
                # use_alt_loss_flag is always False (passed to value_fn if it expects it, but standard fn might not use it)
                # Ensure value_fn handles it or use a wrapper. 
                # _value_fn always accepts flag now.
                dummy_flag = jnp.bool_(False)
                
                v0, g0 = self._v_and_g_adam(theta0, dummy_flag, train_args)

                if self._eval_fn_jit is not None:
                    v_eval_0 = self._eval_fn_jit(theta0, eval_args=eval_args)
                else:
                    v_eval_0 = v0

                loss_log = jnp.full((n_eval,), jnp.nan, dtype=v0.dtype).at[0].set(v_eval_0)
                step_log = jnp.full((n_eval,), -1, dtype=jnp.int32).at[0].set(jnp.int32(0))

                carry = (
                    theta0,
                    opt_state0,
                    jnp.int32(0),
                    v0,
                    jnp.int32(1),
                    loss_log,
                    step_log,
                    jnp.bool_(False),
                    jnp.bool_(jnp.isnan(v0)),
                )

                def cond(c):
                    _, _, step, _, _, _, _, stopped, nan_found = c
                    return (step < max_steps) & (~stopped) & (~nan_found)

                def body(c):
                    theta, opt_state, step, last_eval_loss, eval_idx, loss_log, step_log, stopped, nan_found = c

                    v, g = self._v_and_g_adam(theta, dummy_flag, train_args)

                    updates, opt_state2 = self.optimizer.update(g, opt_state, theta)
                    theta2 = optax.apply_updates(theta, updates)

                    step2 = step + jnp.int32(1)
                    nan2 = nan_found | jnp.isnan(v)
                    do_eval = (step2 % jnp.int32(eval_interval)) == 0

                    def on_eval(_):
                        if self._eval_fn_jit is not None:
                            v_post = self._eval_fn_jit(theta2, eval_args=eval_args)
                        else:
                            v_post = self._value_fn_jit(theta2, dummy_flag, train_args)

                        loss_log2 = loss_log.at[eval_idx].set(v_post)
                        step_log2 = step_log.at[eval_idx].set(step2)
                        eval_idx2 = eval_idx + jnp.int32(1)

                        diff = jnp.abs(last_eval_loss - v_post)
                        stop_now = jnp.bool_(early_stop) & (diff <= jnp.asarray(threshold, v_post.dtype))
                        nan_now = jnp.isnan(v_post)
                        return v_post, eval_idx2, loss_log2, step_log2, stop_now, nan_now

                    def on_no_eval(_):
                        return last_eval_loss, eval_idx, loss_log, step_log, jnp.bool_(False), jnp.bool_(False)

                    last_eval_loss2, eval_idx2, loss_log2, step_log2, stop_now, nan_now = jax.lax.cond(
                        do_eval, on_eval, on_no_eval, operand=None
                    )

                    stopped2 = stopped | stop_now
                    nan2 = nan2 | nan_now

                    return (theta2, opt_state2, step2, last_eval_loss2, eval_idx2, loss_log2, step_log2, stopped2, nan2)

                theta_f, opt_state_f, step_f, last_eval_loss_f, eval_idx_f, loss_log_f, step_log_f, stopped_f, nan_f = \
                    jax.lax.while_loop(cond, body, carry)

                return {
                    "theta": theta_f,
                    "_opt_state": opt_state_f,
                    "steps_run": step_f,
                    "last_eval_loss": last_eval_loss_f,
                    "loss_log": loss_log_f,
                    "step_log": step_log_f,
                    "stopped_early": stopped_f,
                    "nan_found": nan_f,
                }

            return self._jit(solve, static_argnames=("max_steps", "eval_interval", "early_stop"))

    def _build_solve_lbfgs(self):
        def solve(theta0, opt_state0, train_args, eval_args, *, max_steps: int, threshold: float, eval_interval: int, early_stop: bool):
            n_eval = max_steps // eval_interval + 1

            # L-BFGS: always use normal loss (False flag)
            use_alt_loss = jnp.bool_(False)
            
            # v0, g0 from state logic
            # Signature: (params, opt_state, *args, **kwargs) -> (value, grad)
            # Args for fn: use_alt_loss, train_args
            # Must pass state as keyword argument for optax wrapper
            v0, g0 = self._v_and_g_lbfgs(theta0, use_alt_loss, train_args, state=opt_state0)

            if self._eval_fn_jit is not None:
                v_eval_0 = self._eval_fn_jit(theta0, eval_args=eval_args)
            else:
                v_eval_0 = v0

            loss_log = jnp.full((n_eval,), jnp.nan, dtype=v0.dtype).at[0].set(v_eval_0)
            step_log = jnp.full((n_eval,), -1, dtype=jnp.int32).at[0].set(jnp.int32(0))

            carry = (
                theta0,
                opt_state0,
                jnp.int32(0),
                v_eval_0,
                jnp.int32(1),
                loss_log,
                step_log,
                jnp.bool_(False),
                jnp.bool_(jnp.isnan(v0)),
            )

            def cond(c):
                _, _, step, _, _, _, _, stopped, nan_found = c
                return (step < max_steps) & (~stopped) & (~nan_found)

            def body(c):
                theta, opt_state, step, last_eval_loss, eval_idx, loss_log, step_log, stopped, nan_found = c

                # state 재사용 가능 경로
                v, g = self._v_and_g_lbfgs(theta, use_alt_loss, train_args, state=opt_state)

                # L-BFGS update (line search는 value_fn + train_args로 수행)
                def value_fn_lbfgs(p):
                    return self._value_fn_jit(p, use_alt_loss, train_args)

                updates, opt_state2 = self.optimizer.update(
                    g,
                    opt_state,
                    theta,
                    value=v,
                    grad=g,
                    value_fn=value_fn_lbfgs,
                )
                theta2 = optax.apply_updates(theta, updates)
                step2 = step + jnp.int32(1)

                nan2 = nan_found | jnp.isnan(v)

                do_eval = (step2 % jnp.int32(eval_interval)) == 0

                def on_eval(_):
                    # 업데이트 후 loss를 기록/판정(필요 시 1회 forward)
                    if self._eval_fn_jit is not None:
                        v_post = self._eval_fn_jit(theta2, eval_args=eval_args)
                    else:
                        v_post = self._value_fn_jit(theta2, use_alt_loss, train_args)

                    loss_log2 = loss_log.at[eval_idx].set(v_post)
                    step_log2 = step_log.at[eval_idx].set(step2)
                    eval_idx2 = eval_idx + jnp.int32(1)

                    diff = jnp.abs(last_eval_loss - v_post)
                    stop_now = jnp.bool_(early_stop) & (diff <= jnp.asarray(threshold, v_post.dtype))
                    nan_now = jnp.isnan(v_post)
                    return v_post, eval_idx2, loss_log2, step_log2, stop_now, nan_now

                def on_no_eval(_):
                    return last_eval_loss, eval_idx, loss_log, step_log, jnp.bool_(False), jnp.bool_(False)

                last_eval_loss2, eval_idx2, loss_log2, step_log2, stop_now, nan_now = jax.lax.cond(
                    do_eval, on_eval, on_no_eval, operand=None
                )

                stopped2 = stopped | stop_now
                nan2 = nan2 | nan_now

                return (theta2, opt_state2, step2, last_eval_loss2, eval_idx2, loss_log2, step_log2, stopped2, nan2)

            theta_f, opt_state_f, step_f, last_eval_loss_f, eval_idx_f, loss_log_f, step_log_f, stopped_f, nan_f = \
                jax.lax.while_loop(cond, body, carry)

            return {
                "theta": theta_f,
                "_opt_state": opt_state_f,
                "steps_run": step_f,
                "last_eval_loss": last_eval_loss_f,
                "loss_log": loss_log_f,
                "step_log": step_log_f,
                "stopped_early": stopped_f,
                "nan_found": nan_f,
            }

        return self._jit(solve, static_argnames=("max_steps", "eval_interval", "early_stop"))

# Alias for backward compatibility or if it was renamed
POVM_JAX_Trainer = JAX_Debug_Trainer

class HybridGradientTrainer(POVM_JAX_Trainer):
    def __init__(
        self,
        manual_grad_fn: Callable[..., Any],
        param_mask: jnp.ndarray,
        **kwargs
    ):
        """
        Extended Trainer that supports hybrid gradients (AutoDiff + Manual).
        
        Args:
            manual_grad_fn: Callable(theta, *args) -> grad_vector
            param_mask: boolean array (True: use manual grad, False: use auto grad)
            **kwargs: Arguments for POVM_JAX_Trainer
        """
        super().__init__(**kwargs)
        self.manual_grad_fn = manual_grad_fn
        self.param_mask = param_mask
        
        # Re-define _loss_and_grad to use hybrid logic
        def hybrid_loss_fn(theta, *args):
            # 1. Stop gradient for manual parts
            theta_frozen = jax.lax.stop_gradient(theta)
            # If mask is True (manual), use frozen (no grad). If False (auto), use theta (grad).
            theta_eff = jnp.where(self.param_mask, theta_frozen, theta)
            return self.train_cost_fn(theta_eff, *args)

        self._hybrid_value_and_grad = jax.value_and_grad(hybrid_loss_fn, argnums=0)
        
        # We need a wrapper that calls both and mixes them
        def combined_loss_and_grad(theta, *args):
            # Auto-diff part (partial)
            val, grad_auto = self._hybrid_value_and_grad(theta, *args)
            
            # Manual part
            grad_manual = self.manual_grad_fn(theta, *args)
            
            # Combine
            grad_final = jnp.where(self.param_mask, grad_manual, grad_auto)
            
            return val, grad_final

        self._loss_and_grad = combined_loss_and_grad
        
        # Re-compile
        self._loss_and_grad_compiled = self._maybe_jit(self._loss_and_grad)

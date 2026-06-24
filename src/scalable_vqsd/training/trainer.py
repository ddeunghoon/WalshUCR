from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax


jax.config.update("jax_enable_x64", True)


@dataclass
class TrainResult:
    theta: Any
    steps_run: int
    last_eval_loss: float
    stopped_early: bool
    nan_found: bool
    step_log: Any
    loss_log: Any

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
    """JIT-compiled Adam trainer used by the restart-reuse Walsh degree-1 sweep."""

    def __init__(
        self,
        train_cost_fn: Callable[..., jnp.ndarray],
        theta_init: Any,
        optimizer_name: str = "adam",
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.0,
        eval_interval: int = 50,
        enable_x64: bool = True,
        eval_cost_fn: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        if enable_x64:
            jax.config.update("jax_enable_x64", True)

        if str(optimizer_name).lower() != "adam":
            raise ValueError("WalshUCR release trainer supports optimizer_name='adam' only.")

        self.train_cost_fn = train_cost_fn
        self.eval_cost_fn = eval_cost_fn
        self.theta = theta_init
        self.eval_interval = int(eval_interval)
        self.weight_decay = float(weight_decay)

        lr = learning_rate if learning_rate is not None else 1e-2
        self.optimizer = optax.adam(learning_rate=lr)
        self.opt_state = self.optimizer.init(self.theta)

        def _value_fn(params, train_args):
            return self.train_cost_fn(params, *train_args)

        self._value_fn_jit = jax.jit(_value_fn)

        if self.eval_cost_fn is not None:

            def _eval_fn(params, *, eval_args):
                out = self.eval_cost_fn(params, *eval_args)
                if isinstance(out, tuple):
                    return out[0]
                return out

            self._eval_fn_jit = jax.jit(_eval_fn)
        else:
            self._eval_fn_jit = None

        self._value_and_grad = jax.jit(jax.value_and_grad(_value_fn, argnums=0))
        self._solve_adam = self._build_solve_adam()

    def run_optimization(
        self,
        steps: int,
        train_args: tuple,
        eval_args: Optional[tuple] = None,
        threshold: float = 1e-10,
        eval_interval: Optional[int] = None,
        early_stop: bool = True,
        return_numpy_logs: bool = False,
        **kwargs,
    ) -> TrainResult:
        eval_args_pass = eval_args if eval_args is not None else train_args
        result = self._solve_adam(
            self.theta,
            self.opt_state,
            train_args,
            eval_args_pass,
            max_steps=int(steps),
            threshold=float(threshold),
            eval_interval=int(self.eval_interval if eval_interval is None else eval_interval),
            early_stop=bool(early_stop),
            train_tolerance=float(kwargs.get("train_tolerance", 1e-4)),
            switch_step=int(kwargs.get("switch_step", -1)),
        )
        jax.block_until_ready(result["theta"])
        self.theta = result["theta"]
        self.opt_state = result["_opt_state"]

        step_log = result["step_log"]
        loss_log = result["loss_log"]
        if return_numpy_logs:
            import numpy as np

            step_log = np.asarray(step_log)
            loss_log = np.asarray(loss_log)

        return TrainResult(
            theta=result["theta"],
            steps_run=int(result["steps_run"]),
            last_eval_loss=float(result["last_eval_loss"]),
            stopped_early=bool(result["stopped_early"]),
            nan_found=bool(result["nan_found"]),
            step_log=step_log,
            loss_log=loss_log,
        )

    def _build_solve_adam(self):
        def solve(
            theta0,
            opt_state0,
            train_args,
            eval_args,
            *,
            max_steps: int,
            threshold: float,
            eval_interval: int,
            early_stop: bool,
            train_tolerance: float,
            switch_step: int,
        ):
            del train_tolerance, switch_step
            n_eval = max_steps // eval_interval + 1
            v0, _ = self._value_and_grad(theta0, train_args)

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

            def cond(loop_carry):
                _, _, step, _, _, _, _, stopped, nan_found = loop_carry
                return (step < max_steps) & (~stopped) & (~nan_found)

            def body(loop_carry):
                theta, opt_state, step, last_eval_loss, eval_idx, loss_log, step_log, stopped, nan_found = loop_carry
                value, grad = self._value_and_grad(theta, train_args)
                updates, opt_state2 = self.optimizer.update(grad, opt_state, theta)
                theta2 = optax.apply_updates(theta, updates)

                step2 = step + jnp.int32(1)
                nan2 = nan_found | jnp.isnan(value)
                do_eval = (step2 % jnp.int32(eval_interval)) == 0

                def on_eval(_):
                    if self._eval_fn_jit is not None:
                        value_post = self._eval_fn_jit(theta2, eval_args=eval_args)
                    else:
                        value_post = self._value_fn_jit(theta2, train_args)

                    loss_log2 = loss_log.at[eval_idx].set(value_post)
                    step_log2 = step_log.at[eval_idx].set(step2)
                    eval_idx2 = eval_idx + jnp.int32(1)
                    diff = jnp.abs(last_eval_loss - value_post)
                    stop_now = jnp.bool_(early_stop) & (diff <= jnp.asarray(threshold, value_post.dtype))
                    nan_now = jnp.isnan(value_post)
                    return value_post, eval_idx2, loss_log2, step_log2, stop_now, nan_now

                def on_no_eval(_):
                    return last_eval_loss, eval_idx, loss_log, step_log, jnp.bool_(False), jnp.bool_(False)

                last_eval_loss2, eval_idx2, loss_log2, step_log2, stop_now, nan_now = jax.lax.cond(
                    do_eval,
                    on_eval,
                    on_no_eval,
                    operand=None,
                )
                return (
                    theta2,
                    opt_state2,
                    step2,
                    last_eval_loss2,
                    eval_idx2,
                    loss_log2,
                    step_log2,
                    stopped | stop_now,
                    nan2 | nan_now,
                )

            theta_f, opt_state_f, step_f, last_eval_loss_f, _, loss_log_f, step_log_f, stopped_f, nan_f = (
                jax.lax.while_loop(cond, body, carry)
            )
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

        return jax.jit(solve, static_argnames=("max_steps", "eval_interval", "early_stop"))

"""AdaPNM optimizer implementation."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union, Callable, Dict

from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import deserialize
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops


class AdaPNM(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True
    r"""Implements Adaptive Positive-Negative Momentum.
      It has be proposed in 
      'Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve 
      Generalization'
      """

    def __init__(
            self,
            learning_rate: Union[float, Callable, Dict] = 0.001,
            beta_1: Union[float, Callable] = 0.9,
            beta_2: Union[float, Callable] = 0.999,
            beta_3: Union[float, Callable] = 1.0,
            epsilon: float = 1e-8,
            weight_decay: Union[float, Callable, Dict] = 0.0,
            amsgrad: bool = True,
            name='AdaPNM',
            **kwargs):
        super(AdaPNM, self).__init__(name, **kwargs)
        if isinstance(learning_rate, Dict):
            learning_rate = deserialize(learning_rate)
        if isinstance(weight_decay, Dict):
            weight_decay = deserialize(weight_decay)
        self.apply_weight_decay = float(weight_decay) > 0.0
        self._set_hyper('weight_decay', weight_decay)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self._set_hyper('beta_3', beta_3)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'neg_m')
            self.add_slot(var, 'v')
            if self.amsgrad:
                self.add_slot(var, 'vhat')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaPNM, self)._prepare_local(var_device, var_dtype, apply_state)
        weight_decay = array_ops.identity(self._get_hyper('weight_decay', var_dtype))
        epsilon = ops.convert_to_tensor_v2(self.epsilon, var_dtype)
        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_3_t = array_ops.identity(self._get_hyper('beta_3', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr_t = apply_state[(var_device, var_dtype)]['lr_t']
        beta_1_t = math_ops.square(beta_1_t)
        one_minus_beta_1_t = 1.0 - beta_1_t
        one_minus_beta_2_t = 1.0 - beta_2_t
        norm = math_ops.sqrt(1.0 - beta_2_power) / math_ops.maximum(epsilon, 1.0 - beta_1_power)
        noise_norm = math_ops.maximum(epsilon,
                                      math_ops.sqrt(math_ops.square(1.0 + beta_3_t) + math_ops.square(beta_3_t)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=epsilon,
                noise_norm=noise_norm,
                weight_decay=weight_decay,
                lr=lr_t,
                norm=norm,
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                local_step=local_step,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                beta_3_t=beta_3_t,
                one_minus_beta_1_t=one_minus_beta_1_t,
                one_minus_beta_2_t=one_minus_beta_2_t))

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(AdaPNM, self).set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        m = self.get_slot(var, 'm')
        neg_m = self.get_slot(var, 'neg_m')
        beta_m_t = neg_m * coefficients['beta_1_t']
        neg_m_t = state_ops.assign(neg_m, m, use_locking=self._use_locking)
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        m_t = state_ops.assign(m, beta_m_t + m_scaled_g_values, use_locking=self._use_locking)
        pnmomentum = (m_t + (m_t - neg_m_t) * coefficients['beta_3_t']) / coefficients['noise_norm']
        v = self.get_slot(var, 'v')
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = math_ops.square(grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'] + v_scaled_g_values, use_locking=self._use_locking)
        updates = [m_t, v_t, neg_m_t]
        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
            updates.append(v_hat_t)
        else:
            v_hat_t = v_t
        var_t = pnmomentum / math_ops.maximum(math_ops.sqrt(v_hat_t), coefficients['epsilon'])
        var_t *= coefficients['norm']
        if self.apply_weight_decay:
            var_t += coefficients['weight_decay'] * var
        var_update = state_ops.assign_sub(var, coefficients['lr'] * var_t, use_locking=self._use_locking)
        updates.append(var_update)
        return control_flow_ops.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))
        var_slice = array_ops.gather(var, indices)
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, 'm')
        neg_m = self.get_slot(var, 'neg_m')
        beta_m_t = neg_m * coefficients['beta_1_t']
        neg_m_t = state_ops.assign(neg_m, m, use_locking=self._use_locking)
        m_t = state_ops.assign(m, beta_m_t, use_locking=self._use_locking)
        m_scaled_g_values = grad * coefficients['one_minus_beta_1_t']
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        pnmomentum = (m_t + (m_t - neg_m_t) * coefficients['beta_3_t']) / coefficients['noise_norm']
        v = self.get_slot(var, 'v')
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v_scaled_g_values = math_ops.square(grad) * coefficients['one_minus_beta_2_t']
        v_t = state_ops.assign(v, v * coefficients['beta_2_t'], use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
        updates = [m_t, v_t, neg_m_t]
        if self.amsgrad:
            v_hat = self.get_slot(var, 'vhat')
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(
                    v_hat, v_hat_t, use_locking=self._use_locking)
            updates.append(v_hat_t)
        else:
            v_hat_t = v_t
        var_t = pnmomentum / math_ops.maximum(math_ops.sqrt(v_hat_t), coefficients['epsilon'])
        var_t *= coefficients['norm']
        if self.apply_weight_decay:
            var_t += coefficients['weight_decay'] * var_slice
        var_update = state_ops.assign_sub(var, coefficients['lr'] * var_t, use_locking=self._use_locking)
        updates.append(var_update)
        return control_flow_ops.group(*updates)

    def get_config(self):
        config = super(AdaPNM, self).get_config()
        config.update(
            {
                'learning_rate': self._serialize_hyperparameter('learning_rate'),
                'beta_1': self._serialize_hyperparameter('beta_1'),
                'beta_2': self._serialize_hyperparameter('beta_2'),
                'beta_3': self._serialize_hyperparameter('beta_3'),
                'weight_decay': self._serialize_hyperparameter('weight_decay'),
                'epsilon': self.epsilon,
                'amsgrad': self.amsgrad,
            }
        )
        return config

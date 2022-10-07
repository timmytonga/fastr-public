"""
norm version i.e. 𝑎=(1+∑‖∇𝑓(𝑥𝑡,𝜉𝑡)‖/𝑎^2)^(−2/3)
"""
import torch
from .storm_base_class import StormBaseClass


class FastrN(StormBaseClass):
    """
    Extends StormBaseClass by defining specific update rule for Momentum
    """
    def _initialize_momentum(self, state, group):
        """
        Take care of first iter and initialize all the states and momentum
        """
        g = state['g_tilde']  # just for this iter
        per_coordinate = group['per_coordinate']
        dim_normalized = group['dim_normalized']
        # first we set state a_0 to either be the first gradient if a_0 is <= 0 otherwise we use the default value
        state['a_0'] = StormBaseClass.set_proportional_a0(g) if group['a_0'] <= 0 else group['a_0']
        # now we can set everything else
        p, q = group['p'], group['q']
        a_0, b_0 = state['a_0'], group['b_0']
        # eps = group['eps']
        g = state['g_tilde']  # just for this iter
        g_sq = (g*g) if per_coordinate else (g * g).sum()
        d = state['d'] = g
        d_sq = (d*d) if per_coordinate else (d * d).sum()
        if dim_normalized:  # further normalize by param count
            g_sq = g_sq/state['state_params_count']
            d_sq = d_sq/state['state_params_count']
        state['sum_g'] = g_sq
        a = state['a'] = (1 + (state['sum_g'] / a_0)).pow(-2 / 3)
        state['sum_d'] = d_sq
        state['b'] = (torch.pow(b_0 + state['sum_d'], p)) / torch.pow(a, q)

    def _update_momentum_and_lr(self, state, group, g, g_tilde,
                                beta1, beta2, ema, ema_g, ema_d,
                                a_0, b_0, bias_corr1, bias_corr2, eps):
        """
        fastrd specific implementation
        """
        # update momentum and friends here
        p, q = group['p'], group['q']
        per_coordinate = group['per_coordinate']
        dim_normalized = group['dim_normalized']

        state['d'] = g + (1 - state['a']) * (state['d'] - g_tilde)
        d = state['d']
        g_sq = (g*g) if per_coordinate else (g*g).sum()
        d_sq = (d*d) if per_coordinate else (d*d).sum()
        if dim_normalized:  # further normalize by param count
            g_sq = g_sq / state['state_params_count']
            d_sq = d_sq / state['state_params_count']
        state['sum_g'] = beta1 * state['sum_g'] + (1 - beta1) * g_sq if ema or ema_g \
            else state['sum_g'] + g_sq
        state['sum_d'] = beta2 * state['sum_d'] + (1 - beta2) * d_sq if ema or ema_d \
            else state['sum_d'] + d_sq
        state['a'] = (1 + (state['sum_g']/a_0)*bias_corr1).pow(-2/3)
        state['b'] = (torch.pow(b_0 + state['sum_d'] * bias_corr2, p)) / torch.pow(state['a'], q)

import torch
from .storm_base_class import StormBaseClass


class StormPlus(StormBaseClass):
    """
        Extends StormBaseClass by defining specific update rule for Momentum
    """

    def _initialize_momentum(self, state, group):
        """
        Take care of first iter and initialize all the states and momentum
        """
        g = state['g_tilde']  # just for this iter
        per_coordinate = group['per_coordinate']
        # first we set state a_0 to either be the first gradient if a_0 is <= 0 otherwise we use the default value
        state['a_0'] = StormBaseClass.set_proportional_a0(g) if group['a_0'] <= 0 else group['a_0']
        # now we can set everything else
        a_0, b_0 = state['a_0'], group['b_0']
        state['d'] = g
        state['sum_g'] = (g*g) if per_coordinate else (g * g).sum()
        state['a'] = (1 + state['sum_g']/a_0).pow(-2 / 3)
        d, a = state['d'], state['a']
        state['sum_d'] = (d*d)/a if per_coordinate else ((d * d).sum())/a
        state['b'] = torch.pow(b_0 + state['sum_d'], 1/3)

    def _update_momentum_and_lr(self, state, group, g, g_tilde, beta1, beta2, ema, ema_g, ema_d, a_0, b_0, bias_corr1,
                                bias_corr2, eps):
        """
        storm specific implementation
        """
        per_coordinate = group['per_coordinate']
        state['d'] = g + (1 - state['a']) * (state['d'] - g_tilde)
        g_sq = (g*g) if per_coordinate else (g * g).sum()
        state['sum_g'] = beta1 * state['sum_g'] + (1 - beta1) * g_sq if ema or ema_g \
            else (state['sum_g'] + g_sq)
        state['a'] = (1 + (state['sum_g'] / a_0) * bias_corr1).pow(-2 / 3)
        d, a = state['d'], state['a']
        d_sq = d*d if per_coordinate else (d*d).sum()
        state['sum_d'] = beta2 * state['sum_d'] + (1 - beta2) * d_sq / a if ema or ema_d \
            else state['sum_d'] + d_sq / a
        state['b'] = torch.pow(b_0 + state['sum_d'] * bias_corr2, 1 / 3)

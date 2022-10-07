"""
difference of norm version i.e. 𝑎=(1+(1/a_0)∑‖∇𝑓(𝑥𝑖,𝜉𝑖)−∇𝑓(𝑥𝑖,𝜉𝑖+1)‖^2)−2/3
"""
import torch
from .storm_base_class import StormBaseClass


class FastrD(StormBaseClass):
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
        b_0, p = group['b_0'], group['p']
        state['sum_diff_g'] = 0  # initial is 0 so a_1=1
        # then we can update the prev gradient i.e.  ∇𝑓(𝑥_{𝑡-1};𝜉_{𝑡-1}) for next iter
        state['g_prev'] = g.clone().detach()
        # then initialize momentum and friends
        state['a'] = torch.ones_like(g) if per_coordinate else torch.tensor(1)
        state['d'] = g
        d = state['d']
        state['sum_d'] = d*d if per_coordinate else (d * d).sum()
        state['b'] = torch.pow(b_0 + state['sum_d'], p)  # torch.pow((b_0 + state['sum_d']), 1/2)/torch.pow(a, 1/4)

    def _update_momentum_and_lr(self, state, group, g, g_tilde, beta1, beta2, ema, ema_g, ema_d, a_0, b_0, bias_corr1,
                                bias_corr2, eps):
        """
        fastrd specific implementation
        :param eps:
        """
        per_coordinate = group['per_coordinate']
        p, q = group['p'], group['q']
        # first we update the sum of diff
        diff_g = state['g_prev'] - g_tilde
        diff_g_sq = diff_g*diff_g if per_coordinate else (diff_g*diff_g).sum()
        state['sum_diff_g'] = (beta1 * state['sum_diff_g'] + (1 - beta1) * diff_g_sq) if ema or ema_g \
            else (state['sum_diff_g'] + diff_g_sq)
        # then we can update the prev gradient i.e.  ∇𝑓(𝑥_{𝑡-1};𝜉_{𝑡-1}) for next iter
        state['g_prev'] = g.clone().detach()
        # then update momentum and friends
        # state['a'] = torch.minimum(1 / (eps + (state['sum_diff_g'] * bias_corr1 / a_0).pow(2 / 3)), self.min_a)
        state['a'] = (1 + state['sum_diff_g'] * bias_corr1 / a_0).pow(-2 / 3)
        state['d'] = g + (1 - state['a']) * (state['d'] - g_tilde)
        d = state['d']
        d_sq = d*d if per_coordinate else (d*d).sum()
        state['sum_d'] = (beta2 * state['sum_d'] + (1 - beta2) * d_sq) if ema or ema_d \
            else (state['sum_d'] + d_sq)
        state['b'] = (torch.pow(b_0 + state['sum_d'] * bias_corr2, p)) / torch.pow(state['a'], q)

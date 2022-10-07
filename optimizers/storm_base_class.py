import torch.optim as optim
import torch
import numpy as np


class StormBaseClass(optim.Optimizer):
    """
    Abstract class for Storm-based methods. Also provides facility for logging and api for train loop.
    One need to implement (at the minimum) the update momentum functions:
        - _initialize_momentum
        - _update_momentum_and_lr
    These keep track of the states and set the appropriate steps for the step function to be called.
    The step function will save the grad of the current weight/batch and then update the weights using the saved states
        then it is expected in the main training loop for the user to obtain a new grad and called update_momentum
        the update_momentum_and_lr method will set the next states in order for the user to be able to make the next step
    One can reimplement the
    """
    def __init__(self, params, args):
        """
        ema: exponential moving average
        beta1: moving average for sum_g
        beta2: moving average for sum_d
        """
        lr = args.lr
        ema = args.storm_ema
        eps = args.storm_eps
        a_0 = args.storm_a_0
        b_0 = args.storm_b_0
        ema_g = args.storm_ema_g
        ema_d = args.storm_ema_d
        bias_correction = args.storm_bias_correction
        beta1 = args.storm_beta1
        beta2 = args.storm_beta2
        per_coordinate = args.storm_per_coordinate
        p = float(args.storm_p)
        dim_normalized = args.storm_dim_normalized

        assert 0.17 < p <= 0.5, "p must be in (0.17, 1/2]"
        assert lr > 0, "lr must be positive!"
        assert b_0 > 0, "b_0 must be positive!"
        assert not (dim_normalized and per_coordinate), "cannot do per_coord and dim_normalize at the same time"
        assert not (dim_normalized and args.storm_normalized), \
            "cannot do a_0 normalized and dim_normalize at the same time"

        q = (1 - p) / 2
        if ema_g:
            print("ema on sum_g activated")
        if ema_d:
            print("ema on sum_d activated")
        if ema:
            print("ema activated")
        if per_coordinate:
            print("per_coordinate update activated")
        if bias_correction:
            print("bias_correction activated")
        default = dict(lr=lr, a_0=a_0, b_0=b_0, ema=ema, eps=eps, ema_g=ema_g, ema_d=ema_d,
                       beta1=beta1, beta2=beta2, bias_correction=bias_correction,
                       per_coordinate=per_coordinate, p=p, q=q, dim_normalized=dim_normalized)
        super().__init__(params, default)
        if a_0 == -1 and not args.storm_normalized:
            print("a_0 == -1: Making a_0 adaptive to the stochastic gradients")
            assert not args.storm_normalized
        elif a_0 == -2 or args.storm_normalized:
            self.set_a0_num_params()
            print(f"a_0 = {self.param_groups[0]['a_0']}: making a_0 be proportional to the number of parameters")
        elif a_0 <= 0:
            print("a_0 <= 0: Setting a_0 to be a function of the first gradient's norm square")

        self.min_a = torch.tensor(0.999)
        self.step_count = 0
        self.update_momentum_count = 1
        # this flag is important to initialize the first iter correctly
        self.init_flag = False

    def save_g_tilde(self, param):
        grad = param.grad.data
        self.state[param]['g_tilde'] = grad.clone().detach()

    @torch.no_grad()
    def step(self, closure=None):
        for i, group in enumerate(self.param_groups):
            lr = group['lr']  # get this param_group's lr
            for j, param in enumerate(group['params']):
                if param.grad.data is None:
                    continue
                state = self.state[param]
                # first check if we have updated the momentum from last epoch (if this is not the first)
                if ('b' in state) and (state['g_tilde'] is not None):  # ('b' in state) means NOT the first epoch
                    raise Exception("Must update momentum in previous epoch!")
                # save g_tilde for momentum update
                self.save_g_tilde(param)
                # if this is the first update then we update momentum now
                if 'b' not in state:
                    self.initialize_momentum(state, group, param)
                # now we are safe to step
                b, d = state['b'], state['d']
                param.data -= (lr / b) * d  # step and update param
        self.step_count += 1

    def initialize_momentum(self, state, group, param):
        state['state_params_count'] = np.prod(param.shape) if group['dim_normalized'] else None
        self._initialize_momentum(state, group)
        # safety checks below! subtle but crucial.
        state['g_tilde'] = None
        self.init_flag = True  # so that we don't update the momentum again

    def _initialize_momentum(self, state, group):
        # the main difference between algorithms are how the momentum terms are set
        raise NotImplementedError

    @torch.no_grad()
    def update_momentum_and_lr(self):
        """
        This should be called after obtaining ∇𝑓(𝑥_𝑡;𝜉_𝑡) to update d, a, and b
        """
        if self.init_flag:
            self.init_flag = False
            return  # already updated in the init step above

        for i, group in enumerate(self.param_groups):
            b_0, eps = group['b_0'], group['eps']
            ema, ema_g, ema_d = group['ema'], group['ema_g'], group['ema_d']
            beta1, beta2, bias_correction = group['beta1'], group['beta2'], group['bias_correction']
            bias_corr1, bias_corr2 = self.get_bias_correction(beta1, beta2, bias_correction)
            for j, param in enumerate(group['params']):
                g = param.grad.data
                if g is None:
                    continue
                state = self.state[param]
                # this will do nothing if group['a_0'] != -1 but will update to bigger grad
                self.check_and_update_a0(group, state, g)
                g_tilde, a_0 = state['g_tilde'], state['a_0']
                # this update momentum and lr specific to the algorithm
                self._update_momentum_and_lr(state, group, g, g_tilde, beta1, beta2, ema, ema_g, ema_d, a_0, b_0,
                                             bias_corr1, bias_corr2, eps)
                # for safety check: cannot step if momentum is not updated!
                state['g_tilde'] = None

        self.update_momentum_count += 1

    def _update_momentum_and_lr(self, state, group, g, g_tilde, beta1, beta2,
                                ema, ema_g, ema_d, a_0, b_0, bias_corr1,
                                bias_corr2, eps):
        """
        todo: refactor args not needed to simplify loop above
        """

        raise NotImplementedError("Abstract Class!")

    def get_bias_correction(self, beta1, beta2, bias_correction):
        bias_corr1 = 1 / (1 - beta1 ** self.step_count) if bias_correction else 1
        bias_corr2 = 1 / (1 - beta2 ** self.step_count) if bias_correction else 1
        return bias_corr1, bias_corr2

    # Logging stuff below
    def get_params_step_sizes(self, model) -> dict:
        """
        For logging purposes: return a dict of step sizes per param
        :return:
        """
        if len(self.param_groups) != 1:
            raise Exception("len(self.param_groups) not 1! Have to deal with this case later...")
        group = self.param_groups[0]  # 110
        lr = group['lr']  # get this param_group's lr
        per_coordinate = group['per_coordinate']
        result = {}
        for name, p in model.named_parameters():
            if p.grad.data is None:
                continue
            result[name] = self.get_step_size(p, lr, per_coordinate)
        return result

    def get_params_momentum(self, model) -> dict:
        """
        For logging purposes: return a dict of momentum per param
        :return:
        """
        if len(self.param_groups) != 1:
            raise Exception("len(self.param_groups) not 1! Have to deal with this case later...")
        group = self.param_groups[0]  # 110
        per_coordinate = group['per_coordinate']
        result = {}
        for name, p in model.named_parameters():
            if p.grad.data is None:
                continue
            result[name] = self.get_momentum(p, per_coordinate)
        return result

    def get_params_a0(self, model) -> dict:
        """
        For logging purposes: return a dict of momentum per param
        :return:
        """
        if len(self.param_groups) != 1:
            raise Exception("len(self.param_groups) not 1! Have to deal with this case later...")
        group = self.param_groups[0]  # 110
        a_0 = group['a_0']  # get this param_group's lr
        if a_0 != -1:
            return {}
        result = {}
        for name, p in model.named_parameters():
            if p.grad.data is None:
                continue
            result[name] = self.state[p]['a_0']
        return result

    def get_step_size(self, p, lr, per_coordinate) -> int:
        b = self.state[p]['b']
        if per_coordinate:
            return lr/b
        return lr / float(b)

    def get_momentum(self, p, per_coordinate):
        if per_coordinate:
            return self.state[p]['a']
        return float(self.state[p]['a'])

    def get_num_params_of(self):
        num_params = 0
        for group in self.param_groups:
            for p in group['params']:
                num_params += np.prod(p.size())
        return num_params

    def set_a0_num_params(self):
        num_params = self.get_num_params_of()
        for group in self.param_groups:
            group['a_0'] = num_params
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['a_0'] = num_params

    @staticmethod
    def set_proportional_a0(g):
        return (g*g).sum()

    @classmethod
    def check_and_update_a0(cls, group, state, g):
        """
        Since we want to set a_0 proportional to G (the bound on the stochastic gradients),
        we will update G and hence update a_0 here
        """
        a_0 = state['a_0']
        if group['a_0'] == -1 and cls.set_proportional_a0(g) > a_0:
            state['a_0'] = cls.set_proportional_a0(g)
            return state['a_0']

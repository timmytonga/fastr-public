import torch


def custom_noncvx_loss(output: torch.Tensor, target: torch.Tensor):
    """
    from Section F of https://arxiv.org/pdf/1810.10690.pdf
    """
    return torch.mean(torch.log(torch.square(target - output)/2 + 1))


def custom_noncvx_reg_loss(weight, base_loss, reg, alpha):
    def custom_loss(output, target):
        return base_loss(output, target) + reg(weight, alpha)
    return custom_loss


def custom_noncvx_reg(weight: torch.Tensor, alpha):
    w_sq = torch.square(weight)
    return alpha * torch.sum(w_sq/(1+w_sq))

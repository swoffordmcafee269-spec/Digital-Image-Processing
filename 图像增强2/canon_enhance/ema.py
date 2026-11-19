import torch


class ModelEMA:
    """EMA of trainable weights; fixed calibration buffers remain exact."""
    def __init__(self,model,decay=.999,state=None):
        self.decay=decay
        self.state={k:v.detach().clone() for k,v in model.state_dict().items()}
        self.parameter_names=[k for k,_ in model.named_parameters()]
        if state:
            for k,v in state.items():self.state[k].copy_(v)

    @torch.no_grad()
    def update(self,model):
        live=dict(model.named_parameters())
        shadow=[self.state[k] for k in self.parameter_names]
        current=[live[k].detach() for k in self.parameter_names]
        torch._foreach_mul_(shadow,self.decay)
        torch._foreach_add_(shadow,current,alpha=1-self.decay)

    def apply(self,model):
        backup={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.state)
        return backup

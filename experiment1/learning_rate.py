
#schedular=StepLR(optimizer,20,gamma=0.1)
from torch.optim.optimizer import Optimizer

class _LRSchedular(object):
    def __init__(self,optimizer,last_epoch=-1):
        if not isinstance(optimizer,Optimizer):
            raise TypeError('{}is not an Optimizer'.format(type(optimizer).__name__))
        self.optimzer=optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr',group['lr'])
        else:
            for i,group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified in /"
                                   "param_groups[{}] when resuming an optimizer".format(i))

        self.step(last_epoch+1)
        self.last_epoch=last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self,epoch=None):
        if epoch is None:
            epoch=self.last_epoch+1
        self.last_epoch=epoch
        for param_group,lr in zip(self.optimzer.param_groups,self.get_lr()):
            param_group['lr']=lr



class StepLR(_LRSchedular):
    def __init__(self,optimizer,step_size,gamma=0.1,last_epoch=-1):
        self.step_size=step_size
        self.gamma=gamma
        self.base_lrs=[0.1]
        super(StepLR, self).__init__(optimizer,last_epoch)

    def get_lr(self):
        return [base_lr*self.gamma**(self.last_epoch//self.step_size)
                for base_lr in self.base_lrs]

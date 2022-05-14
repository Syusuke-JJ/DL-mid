from pathlib import Path
from means import cutmix,mixup
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.cuda
import torch
from tqdm.auto import tqdm


class Trainer(object):
    cuda=torch.cuda.is_available()
    torch.backends.cudnn.benchmark=True

    def __init__(self,model,optimizer,save_dir=None,save_freq=10):
        self.model=model
        if self.cuda:
            model.cuda()
        self.optimizer=optimizer
        self.save_dir=save_dir
        self.save_freq=save_freq
        self.loss_f=nn.CrossEntropyLoss()

    def _iteration(self,name,data_loader,writer,train_loss,ep,is_train=True):
        loop_loss=[]
        accuracy=[]
        for data,target in tqdm(data_loader,ncols=8):
            if self.cuda:
                data,target=data.cuda(),target.cuda()

            mixed_x, y_a, y_b, lam = eval(name)(data, target)

            output = self.model(mixed_x)
            loss = lam* self.loss_f(output, y_a) + (1 - lam) * self.loss_f(output, y_b)

            loop_loss.append(loss.data.item()/len(data_loader))
            accuracy.append((output.data.max(1)[1]==target.data).sum().item())
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            l=loss.data.item()/len(data_loader)
            train_loss.append(l)
            writer.add_scalar('train_loss',l,ep)
        mode="train"
        print("[{"+mode+"}]loss:("+str(sum(loop_loss))+"}/accuracy:{"+str()+"}")
        print()
        return loop_loss,accuracy

    def test_iteration(self,data_loader,writer,test_loss,test_acc,ep):
        loop_loss=[]
        accuracy=[]
        for data,target in tqdm(data_loader,ncols=8):
            if self.cuda:
                data,target=data.cuda(),target.cuda()
            output=self.model(data)
            loss=self.loss_f(output,target)
            loop_loss.append(loss.data.item()/len(data_loader))
            accuracy.append((output.data.max(1)[1]==target.data).sum().item())
            test_loss.append(loss.item())
            acc=(output.data.max(1)[1]==target.data).sum().item()
            test_acc.append(acc)
            writer.add_scalar('test_loss',loss,ep)
            writer.add_scalar('test_accuracy',acc,ep)
        mode="test"
        print("[{"+mode+"}]loss:("+str(sum(loop_loss))+"}/accuracy:{"+str()+"}")
        print()
        return loop_loss,accuracy

    def train(self,name,data_loader,writer,train_loss,ep):
        self.model.train()
        with torch.enable_grad():
            loss,correct=self._iteration(name,data_loader,writer,train_loss,ep)

    def test(self,data_loader,writer,test_loss,test_acc,ep):
        self.model.eval()
        with torch.no_grad():
            loss,correct=self.test_iteration(data_loader,writer,test_loss,test_acc,ep)

    def loop(self,name,epochs,train_data,test_data,scheduler=None):
        writer=SummaryWriter('tensorboard/'+name)
        train_loss=[]
        test_loss=[]
        test_acc=[]
        for ep in range(epochs):
            if scheduler is not None:
                scheduler.step()
            print("epochs:{}".format(ep))
            self.train(name,train_data,writer,train_loss,ep)
            self.test(test_data,writer,test_loss,test_acc,ep)
            if ep%self.save_freq:
                self.save(ep)
        return train_loss,test_loss,test_acc

    def save(self,epoch,**kwargs):
        if self.save_dir is not None:
            model_out_path=Path(self.save_dir)
            state={"epoch":epoch,"weight":self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state,model_out_path/"model_epoch_{}.pth".format(epoch))
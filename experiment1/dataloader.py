from PIL.Image import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

transform=transforms.Compose([transforms.Resize((256,256)),
                              transforms.RandomCrop(224),
                              transforms.ToTensor(),
                              transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self,images,labels,loader=default_loader,transform=None):
        self.images=images
        self.labels=labels
        self.loader=loader
        self.transform=transform

    def __getitem__(self, index):  #返回tensor
        img,target=self.images[index],self.labels[index]
        img=self.loader(img)
        if self.transform is not None:
            img=self.transform(img)
        return img,target

    def __len__(self):
        return len(self.images)
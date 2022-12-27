from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    
    def __init__(self, root_dir, label_dir):  # root_dir指定到train之前的文件夹,label_dir即为label_name
        super().__init__()
        self.root_dir = root_dir  # 将创建对象的时候给对象的初始路径给到这个对象的全局
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)  # 拼接
        self.img_path_list = os.listdir(self.path)  # 获取list

        
    def __getitem__(self, idx):
        # 获取每一个图片
        img_name = self.img_path_list[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        
        return img, label  # 真的妙
        
    def __len__(self):
        
        return len(self.img_path_list)  # train/ants里面的长度
        
    
# 定义一个数据类

ants_dataset = MyData('data\\hymenoptera_data\\train', 'ants')
bees_dataset = MyData('data\\hymenoptera_data\\train', 'bees')
ants_dataset[0]  # 当类中有__getitem__方法的时候，实例化对象A后调用A[idx]会自动调用__getitem__方法
img, label = ants_dataset[0]

bees_dataset[1][0].show() # 即第二张图img.show()

train_datasets = ants_dataset + bees_dataset  # 牛逼 
print(len(ants_dataset), len(bees_dataset), len(train_datasets))


"""另一种数据集读取 https://www.bilibili.com/video/BV1hE411t7RN?t=1482.2&p=7"""
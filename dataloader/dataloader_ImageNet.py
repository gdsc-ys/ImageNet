# import dataset structuring and image transformation modules
import torch
from torchvision.datasets import ImageFolder  # Load dataset
from torchvision import transforms
import os

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data/imagenette2-320"
)  # use relative directory path
MINI_BATCH_SIZE = 256  # numbers of images in a mini-batch
VALID_BATCH_SIZE = 64  # numbers of images in a mini-batch
LEARNING_RATE = 1e-4
EPOCHS = 100
RESOLUTION = (224, 224)

imagenet_transforms = transforms.Compose(
    [
        transforms.Resize(RESOLUTION),  # 3(channel) x 224(width) x 224(height) Resizing
        transforms.ToTensor(),  # np.array or float -> torch.Tensor()
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


train_data = ImageFolder(root=f"{DATA_PATH}/train", transform=imagenet_transforms)
print("number of training data: ", len(train_data))

valid_data = ImageFolder(root=f"{DATA_PATH}/val", transform=imagenet_transforms)
print("number of test data: ", len(valid_data))

# using torch dataloader to divide dataset into mini-batch
# https://pytorch.org/docs/stable/data.html
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=MINI_BATCH_SIZE, shuffle=True
)
eval_loader = torch.utils.data.DataLoader(
    dataset=valid_data, batch_size=VALID_BATCH_SIZE, shuffle=True
)

first_batch = train_loader.__iter__().__next__()
print("{:<21s} | {:<25s} | {}".format("name", "type", "size"))
print("{:<21s} | {:<25s} | {}".format("Number of Mini-Batchs", "", len(train_loader)))
print("{:<21s} | {:<25s} | {}".format("first_batch", str(type(first_batch)), len(first_batch)))
print(
    "{:<21s} | {:<25s} | {}".format(
        "first_batch[0]", str(type(first_batch[0])), first_batch[0].shape
    )
)
print(
    "{:<21s} | {:<25s} | {}".format(
        "first_batch[1]", str(type(first_batch[1])), first_batch[1].shape
    )
)

################################################################################################################
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

### Dataloader 작성
##
def make_datapath_list():

    train_img_list = list()

    for img_idx in range(200):
        img_path = f"{DATA_PATH}/train/n01440764" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

        img_path = f"{DATA_PATH}/train/n2102040" + str(img_idx) + ".jpg"
        train_img_list.append(img_path)

    return train_img_list


## 이미지 전처리 클래스
class ImageTransform:
    """
    __init__은 객체 생성될 때 불러와짐 / __call__은 인스턴스 생성될 때 불러와짐
    __call__함수는 이 클래스의 객체가 함수처럼 호출되면 실행되는 함수임...
    """

    def __init__(self):
        self.data_tranform = transforms.Compose(
            [
                transforms.Resize(RESOLUTION),  # 3(channel) x 224(width) x 224(height) Resizing
                transforms.ToTensor(),  # np.array or float -> torch.Tensor()
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img):
        return self.data_transform(img)


## 이미지 데이터셋 클래스, pytorch 데이터셋 클래스 상속
class Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        # 이미지 갯수 리턴
        return len(self.file_list)

    def __getitem__(self, index):
        # 전처리 한 이미지의 텐서 형태 취득
        img_path = self.file_list[index]
        img = Image.open(img_path)
        # 이미지 전처리
        img_transformed = self.transform(img)

        return img_transformed

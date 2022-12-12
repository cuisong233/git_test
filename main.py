import traceback
import inspect
import yaml
import sys
import os
import tqdm
import datetime
import uuid
import torchvision
import model
import torch
import torch.optim as optim
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=runs --port=5200

writer = SummaryWriter('runs/fashion_mnist_experiment_1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ['VAR_NAME']=sys.argv[1]                       # 设置与实验相关的环境变量（如CUDA_VISIBLE_DEVICES）
experiment_id = str(uuid.uuid1())[:8]                    # 生成本次实验的UUID
experiment_name = '初始版本'                                   # 设置描述本次实验的名称
logger=dict()                                            # 用字典保存代码、进程ID、配置参数、开始时间、训练时产生的数据等日志信息
logger['experiment_id'] = experiment_id                  # 保存本次实验的UUID
logger['experiment_name'] = experiment_name              # 保存本次实验的名称
logger['code']=inspect.getsource(sys.modules[__name__])  # 保存本次实验代码
logger['pid']=os.getpid()                                # 保存本次实验进程PID
logger['config']= "??"                                  # 保存配置参数
logger['datetime']=str(datetime.datetime.now())          # 保存训练开始时间
logger['loss'] = []                                      # 保存loss日志
logger['info'] = []                                      # 保存其他日志信息
logger['env_vars'] = os.environ                          # 保存相关环境变量
batch_cnt = 0
log_freq = 100

""" 超参数 """
LR = 0.001
EPOCHS = 100
BATCHSIZE = 16


""" 数据准备 """
# 定义 数据集的变换
img_transform = transforms.Compose([
    transforms.Resize(64),  # 调整到 64*64 大小
    transforms.ToTensor(),  # 转为tensor
    transforms.Normalize(mean=0.5, std=0.5)
]   )
# 设置mnist数据
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=img_transform
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCHSIZE, shuffle=True, drop_last=True)


""" 模型准备 """
model = model.Net().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)





""" 训练 """
try:
    for epoch in range(EPOCHS):
        train_bar = tqdm.tqdm(train_loader)
        for step, (x, y) in enumerate(train_bar):
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger['loss'].append(loss)
            # logger['info'].append(info)
            batch_cnt += 1
            train_bar.set_description(desc=f"[{epoch}/{epoch - 1}] [Train] [Loss : {loss:.4f},")

            if batch_cnt % log_freq == 0: # 每log_freq个batch保存一次日志
                writer.add_scalar("training loss", loss, epoch * len(train_loader) + step)
                with open(experiment_name + experiment_id + '.log','w') as f:
                    f.write(yaml.dump(logger, Dumper=yaml.CDumper)) # 使用yaml保存日志
except KeyboardInterrupt:
    print('manully stop training...')
except Exception:
    print(traceback.format_exc())
finally:
    pass
    # postprocess(model) # 训练结束后处理部分，比如保存模型权重等信息到磁盘

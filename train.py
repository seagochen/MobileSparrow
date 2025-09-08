"""
@Fire
https://github.com/fire717
"""
from lib import init, Data, MoveNet, Task

from config import cfg



def main(cfg):

    init(cfg)

    # 创建模型
    # model = MoveNet(num_classes=cfg["num_classes"],
    #                 width_mult=cfg["width_mult"],
    #                 mode='train')
    # print(model)
    # b
    model = MoveNet(
        num_joints=cfg["num_classes"],
        width_mult=cfg["width_mult"]
    )

    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()
    # data.showData(train_loader)
    # b


    run_task = Task(cfg, model)
    run_task.train(train_loader, val_loader)




if __name__ == '__main__':
    main(cfg)
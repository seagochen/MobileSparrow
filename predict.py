"""
@Fire
https://github.com/fire717
"""

from config import cfg
from lib import init, Data, MoveNet, Task


def main(cfg):

    init(cfg)

    model = MoveNet(
        num_joints=cfg["num_classes"],
        width_mult=cfg["width_mult"]
    )

    # model = MoveNet(num_classes=cfg["num_classes"],
    #                 width_mult=cfg["width_mult"],
    #                 mode='train')
    
    data = Data(cfg)
    test_loader = data.getTestDataloader()

    run_task = Task(cfg, model)
    run_task.modelLoad("output/best.pt")

    run_task.predict(test_loader, "output/predict")



if __name__ == '__main__':
    main(cfg)
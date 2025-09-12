# from cli.base_cli import BaseCLI
# from core.models.ssdlite import SSDLiteDet
# from core.datasets.coco_loader import CoCo2017DataLoader
# from core.task.det_trainer import DetTrainer
#
# class SSDLiteCLI(BaseCLI):
#     def build_model(self, cfg, num_classes):
#         return SSDLiteDet(
#             num_classes=num_classes, # SSDLite 需要 num_classes (背景+N)
#             # ... 其他 SSDLiteDet 参数 ...
#         )
#
#     def build_data(self, cfg):
#         data = CoCo2017DataLoader(cfg, task="det")
#         train_loader, val_loader = data.getTrainValDataloader()
#         # det任务的类别数对模型至关重要
#         num_classes = data.num_classes
#         return {"train": train_loader, "val": val_loader}, num_classes
#
#     def get_trainer_class(self):
#         return DetTrainer
#
# if __name__ == "__main__":
#     cli = SSDLiteCLI()
#     cli.run()
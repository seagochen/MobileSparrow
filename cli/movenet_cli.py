# from cli.base_cli import BaseCLI
# from core.models.movenet import MoveNet
# from core.datasets.coco_loader import CoCo2017DataLoader
# from core.task.kpts_trainer import KptsTrainer
#
# class MoveNetCLI(BaseCLI):
#     def build_model(self, cfg, num_classes):
#         # MoveNet的 num_classes 来自 num_joints
#         num_joints = cfg.get("task_params", {}).get("num_joints", 17)
#         return MoveNet(
#             num_joints=num_joints,
#             width_mult=cfg.get("width_mult", 1.0),
#             # ... 其他 MoveNet 参数 ...
#         )
#
#     def build_data(self, cfg):
#         data = CoCo2017DataLoader(cfg, task="kpts")
#         train_loader, val_loader = data.getTrainValDataloader()
#         # kpts任务的类别数通常不影响模型结构，可以返回None或num_joints
#         num_classes = cfg.get("task_params", {}).get("num_joints", 17)
#         return {"train": train_loader, "val": val_loader}, num_classes
#
#     def get_trainer_class(self):
#         return KptsTrainer
#
#     def add_subparsers_hook(self, subparsers):
#         # 为 export-onnx 命令添加 kpts 任务专属的 --keypoints 参数
#         sp_ex = subparsers.choices.get("export-onnx")
#         if sp_ex:
#             sp_ex.add_argument("--keypoints", action="store_true",
#                                help="导出关键点版 ONNX (单一输出 [B,51])")
#
# if __name__ == "__main__":
#     cli = MoveNetCLI()
#     cli.run()
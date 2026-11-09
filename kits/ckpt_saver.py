import os
import torch


class StateDictSaver(object):
    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(
            args.save_dir,
            'weights',
            args.model,
            args.dataset_name,
            args.run_id,
        )
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_epoch_weights(self, state_dict, epoch):
        weight_name = f"{self.args.model}-epoch{epoch:03d}.pth"
        weight_path = os.path.join(self.directory, weight_name)
        torch.save(state_dict, weight_path)
        return weight_path

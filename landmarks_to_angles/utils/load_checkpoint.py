import collections

import torch


def load_checkpoint2(model, checkpoint):
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)["model"]
    source_state_ = checkpoint
    source_state = {}

    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()

    for k in source_state_:
        if k.startswith('module') and not k.startswith('module_list'):
            source_state[k[7:]] = source_state_[k]
        else:
            source_state[k] = source_state_[k]

    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Cannot found pre-trained parameters for {}'.format(target_key))

    model.load_state_dict(new_target_state, strict=False)

    return model


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)["model"]
    source_state_ = checkpoint
    source_state = dict()
    target_state = model.state_dict()
    new_target_state = collections.OrderedDict()

    for k in source_state_:
        if k.startswith('module') and not k.startswith('module_list'):
            source_state[k[7:]] = source_state_[k]
        else:
            source_state[k] = source_state_[k]

    for target_key, target_value in target_state.items():
        if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    model.load_state_dict(new_target_state, strict=False)
    return model

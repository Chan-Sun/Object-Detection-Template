#%%
#### print whole config file
import os
from mmengine import Config, DictAction
from mmdet.utils import replace_cfg_vals, update_data_root
cfg_path = ""
save_path = ""
cfg = Config.fromfile(cfg_path)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)
# update data root according to MMDET_DATASETS
update_data_root(cfg)

print(f'Config:\n{cfg.pretty_text}')

if save_path is not None:
    suffix = os.path.splitext(save_path)[-1]
    assert suffix in ['.py', '.json', '.yml']
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    cfg.dump(save_path)
    print(f'Config saving at {save_path}')

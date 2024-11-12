import torch
import torch.nn as nn
from utils.config import get_config
from utils.logger import create_logger
import time
import numpy as np
from utils.config import get_config
from trainers import vificlip
from datasets.pipeline import *

### Set values here ###
config = "configs/zero_shot/train/k400/16_16_vifi_clip.yaml"
output_folder_name = "exp"
pretrained_model_path = (
    "/Users/benjamin/Downloads/vifi_clip_10_epochs_k400_full_finetuned.pth"
)
# List the action names for which ViFi-CLIP will perform action recognition
class_names = ["dancing", "drum beating", "swimming", "climbing stairs"]
# Load your video example:
video_path = (
    "/Users/benjamin/Downloads/Walking_up_the_stairs_climb_stairs_l_cm_np1_ba_med_1.avi"
)


# Step 1:
# Configuration class
class parse_option:
    def __init__(self):
        self.config = config
        self.output = (
            output_folder_name  # Name of output folder to store logs and save weights
        )
        self.resume = pretrained_model_path
        # No need to change below args.
        self.only_test = True
        self.opts = None
        self.batch_size = None
        self.pretrained = None
        self.accumulation_steps = None
        self.local_rank = 0


args = parse_option()
config = get_config(args)
# logger
logger = create_logger(output_dir=args.output, name=f"{config.MODEL.ARCH}")
logger.info(f"working dir: {config.OUTPUT}")

# Step 2:
# Create the ViFi-CLIP models and load pretrained weights
model = vificlip.returnCLIP(
    config,
    logger=logger,
    class_names=class_names,
)
model = model.float().cuda()  # changing to cuda here


logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")
load_state_dict = checkpoint["model"]
# now remove the unwanted keys:
if "module.prompt_learner.token_prefix" in load_state_dict:
    del load_state_dict["module.prompt_learner.token_prefix"]

if "module.prompt_learner.token_suffix" in load_state_dict:
    del load_state_dict["module.prompt_learner.token_suffix"]

if "module.prompt_learner.complete_text_embeddings" in load_state_dict:
    del load_state_dict["module.prompt_learner.complete_text_embeddings"]
# create new OrderedDict that does not contain `module.`
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in load_state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v

# load params
msg = model.load_state_dict(new_state_dict, strict=False)
logger.info(f"resume model: {msg}")

# Step 3:
# Preprocessing for video
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False
)
scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)
val_pipeline = [
    dict(type="DecordInit"),
    dict(
        type="SampleFrames",
        clip_len=1,
        frame_interval=1,
        num_clips=config.DATA.NUM_FRAMES,
        test_mode=True,
    ),
    dict(type="DecordDecode"),
    dict(type="Resize", scale=(-1, scale_resize)),
    dict(type="CenterCrop", crop_size=config.DATA.INPUT_SIZE),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="FormatShape", input_format="NCHW"),
    dict(type="Collect", keys=["imgs"], meta_keys=[]),
    dict(type="ToTensor", keys=["imgs"]),
]
if config.TEST.NUM_CROP == 3:
    val_pipeline[3] = dict(type="Resize", scale=(-1, config.DATA.INPUT_SIZE))
    val_pipeline[4] = dict(type="ThreeCrop", crop_size=config.DATA.INPUT_SIZE)
if config.TEST.NUM_CLIP > 1:
    val_pipeline[1] = dict(
        type="SampleFrames",
        clip_len=1,
        frame_interval=1,
        num_clips=config.DATA.NUM_FRAMES,
        multiview=config.TEST.NUM_CLIP,
    )
pipeline = Compose(val_pipeline)

dict_file = {"filename": video_path, "tar": False, "modality": "RGB", "start_index": 0}

video = pipeline(dict_file)
video_tensor = video["imgs"].unsqueeze(0).cuda().float()
# Inference through ViFi-CLIP
with torch.no_grad():
    with torch.cuda.amp.autocast():
        logits = model(video_tensor)
pred_index = logits.argmax(-1)

print(f"logits: {logits}")
print(f"predicted action category is : {class_names[pred_index]}")

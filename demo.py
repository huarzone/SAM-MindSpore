import numpy as np
import mindspore as ms

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry


if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](
        checkpoint="/root/SAM-Adapter/weights/sam_pretrain/sam_vit_b_01ec64.ckpt"
    )
    demo_image = ms.Tensor(np.random.rand(1, 3, 1024, 1024).astype(np.float32))
    print(sam.image_encoder(demo_image))
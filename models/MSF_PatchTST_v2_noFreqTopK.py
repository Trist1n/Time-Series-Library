from models.MSF_PatchTST_v2 import MSFPatchTSTV2Base


class Model(MSFPatchTSTV2Base):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(
            configs,
            patch_len=patch_len,
            stride=stride,
            use_freq_topk=False,
            use_decomp=True,
            use_scale_fusion=True,
        )

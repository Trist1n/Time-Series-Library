from models.MSF_PatchTST_v3 import MSFPatchTSTV3Base


class Model(MSFPatchTSTV3Base):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__(
            configs,
            patch_len=patch_len,
            stride=stride,
            use_period=False,
            use_channel_graph=True,
            use_spectral_gate=True,
        )

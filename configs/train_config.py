"""
Training configuration for the JoImTeRNet-based sentence-level retrieval
model (manuscript Section 3.1.1).

All hyperparameters, seeds, and paths used in the reported experiments are
set here. `cfg` is a module-level singleton instance, imported by
`src/engine/trainer.py`, `src/losses.py`, and `training/train.py` — this
mirrors how the original single-file script used one global `cfg` object.
"""


class Config(object):
    def __init__(self):
        self.task = 'JoImTeR sentence-level training on MIMIC-CXR 20k'

        self.scheduler_init = True
        self.scheduler_step = False

        self.multitasksampling = 0.7  # P(ITM step) vs P(MLM step) per training step

        self.DATASET_NAME = 'MIMIC'
        self.CUDA = True
        self.snapshot_interval = 1

        # Path to a text-encoder checkpoint to resume/initialize from (e.g. an
        # earlier MLM-pretraining run). Set to '' to train the text encoder
        # from scratch. The path below is the Kaggle path used for the
        # reported run — replace with your own checkpoint path or ''.
        self.text_encoder_path = (
            '/kaggle/input/notebooks/shery432/mimic-rag/output/'
            'MIMIC_sentence_level_2026_07_03_12_02_10/Model/text_encoder_latest.pth'
        )
        self.joint_encoder_path = ''

        self.CONFIG_NAME = 'sentence_level'
        self.DATA_DIR = '../'
        self.TRAIN = True
        self.GPU_ID = 0

        # Word-region alignment (WRA) attention/similarity temperatures and
        # sentence-level matching temperature (JoImTeRNet formulation).
        self.GAMMA1 = 4.0
        self.GAMMA2 = 5.0
        self.GAMMA3 = 10.0
        self.sent_margin = 0.3
        self.word_margin = 0.3
        self.LAMBDA_DAMSM = 2.0
        self.LAMBDA_TRIPLET = 1.0

        self.clip_max_norm = 1.5

        self.max_length = 128
        self.hidden_dim = 512

        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1

        self.lr = 1e-4
        self.weight_decay = 1e-4

        self.epochs = 30
        self.lr_drop = 15
        self.lr_gamma = 0.1
        self.start_epoch = 0

        self.seed = 42
        self.batch_size = 24
        self.val_batch_size = 16
        self.num_workers = 4

        self.pretrained = True
        self.freeze_backbone = False
        self.lr_backbone = 1e-5


# Module-level singleton, imported as `from configs.train_config import cfg`
# throughout src/ and training/.
cfg = Config()

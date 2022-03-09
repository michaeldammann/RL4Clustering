DATASET = "MNIST"
BATCH_SIZE = 1024
FEATURE_DIM = 16
LR = 1e-4
EPOCHS = 25
REWARD_CLASSES = 1

SAVE_INTERVALL = 1
SAVE_DIR = 'result_batch_size_{}_feature_dim_{}_reward_classes_{}_learningrate_{}'.format(str(BATCH_SIZE), str(FEATURE_DIM),
                                                                          str(REWARD_CLASSES), str(LR))

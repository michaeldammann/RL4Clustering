DATASET = "MNIST"
BATCH_SIZE = 4096
FEATURE_DIM = 2
LR = 1e-2
EPOCHS = 200
REWARD_CLASSES = 1

SAVE_INTERVALL = 1
SAVE_DIR = 'result_batch_size_{}_feature_dim_{}_reward_classes_{}_learningrate_{}_rew_nn'.format(str(BATCH_SIZE), str(FEATURE_DIM),
                                                                          str(REWARD_CLASSES), str(LR))

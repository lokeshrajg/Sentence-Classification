config = {}


class TrainingConfig(object):
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 30
    decay_rate = 0.95
    epochs = 10
    evaluate_every = 1
    checkpoint_every = 1


class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
    fully_connected_layers = [512, 512]
    th = 1e-6
    embedding_size = 64


class Config(object):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet_size = len(alphabet)
    l0 = 512
    batch_size = 128
    num_of_classes = 12
    dropout_p = 0.5
    train_data_source = 'data/xtrain_obfuscated.txt'
    test_data_source = 'data/xtest_obfuscated.txt'
    train_label_source = 'data/ytrain.txt'
    test_label_source = None
    training = TrainingConfig()
    model = ModelConfig()

config = Config()
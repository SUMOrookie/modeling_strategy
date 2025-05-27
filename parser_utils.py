import argparse

def get_parser(flag):
    parser = argparse.ArgumentParser()
    if flag == "train":
        parser.add_argument('--seed', type=int, default=47, help='Random seed.')
    elif flag == "test":
        parser.add_argument('--seed', type=int, default=996, help='Random seed.')
    else:
        raise Exception("")
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')

    # parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train.')
    parser.add_argument('--epochs', type=int, default=999, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
    # parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=6, help='Number of head attentions.')
    # parser.add_argument('--nb_heads', type=int, default=16, help='Number of head attentions.')
    # parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha for the leaky_relu.')
    # parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=300, help='Patience')
    return parser
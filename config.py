import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch TreeLSTM for Sentence Similarity on Dependency Trees')
    #
    parser.add_argument('--data', default='data/',
                        help='path to dataset')
    parser.add_argument('--glove', default='data/glove/',
                        help='directory with GLOVE embeddings')
    parser.add_argument('--save', default='checkpoints/',
                        help='directory to save checkpoints in')
    parser.add_argument('--data_dir', type=str, default='test',
                        help='Name to identify experiment')
    
    #parser.add_argument('--val_data', type=str,
    #                    help='val data dir')
    # model arguments
    #parser.add_argument('--input_dim', default=300, type=int,
    #                    help='Size of input word vector')
    #parser.add_argument('--mem_dim', default=150, type=int,
    #                    help='Size of TreeLSTM cell state')
    #parser.add_argument('--hidden_dim', default=50, type=int,
    #                    help='Size of classifier MLP')
    #parser.add_argument('--num_classes', default=5, type=int,
    #                    help='Number of classes in dataset')
    parser.add_argument('--freeze_emb', action='store_true',
                        help='Freeze word embeddings')
    # training arguments
    parser.add_argument('--epochs', default=20, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batchsize', default=25, type=int,
                        help='batchsize for optimizer updates')
    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    
    
    parser.add_argument('--mean_only', action='store_true',
                        help='mean-mean structure')

    parser.add_argument('--pretrain', action='store_true',
                        help='load pretrained embedding weights')
    parser.add_argument('--seed', default=123, type=int,
                        help='random seed (default: 123)')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    #cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    #cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')
    #parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args

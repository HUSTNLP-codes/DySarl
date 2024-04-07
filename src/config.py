import argparse

parser = argparse.ArgumentParser(description="Knowledge Graph Embedding")
parser.add_argument("--dataset", "--d", type=str, default="DEBUG", choices=["FBIMG", "WN9IMG"],
                    help="Knowledge Graph dataset")
parser.add_argument("--model", choices=['two', 'twoe'], default="two",
                    help="Knowledge Graph embedding models")
parser.add_argument("--debug", action="store_true",
                    help="Only use 1000 examples for debugging")
parser.add_argument("--double-precision", action="store_true",
                    help="Machine precision")
parser.add_argument("--regularizer", choices=[None, "N3", "F2"], default=None,
                    help="Regularizer")
parser.add_argument("--optimizer", choices=["Adagrad", "Adam"], default="Adam",
                    help="Optimizer")
parser.add_argument("--max-epochs", default=200, type=int,
                    help="Maximum number of epochs to train for")
parser.add_argument("--patience", default=100, type=int,
                    help="Number of epochs before early stopping")
parser.add_argument("--valid-freq", default=1, type=int,
                    help="Number of epochs before validation")
parser.add_argument("--batch-size", default=1000, type=int,
                    help="Batch size")
parser.add_argument("--neg-sample-size", default=0, type=int,
                    help="Negative sample size, 0 not  use negative sampling")
parser.add_argument("--double-neg", action="store_true",
                    help="Whether to negative sample both head and tail entities")
parser.add_argument("--init-size", default=1e-3, type=float,
                    help="Initial embeddings' scale")
parser.add_argument("--learning-rate", default=1e-3, type=float,
                    help="Learning rate")
parser.add_argument("--multi-c", action="store_true",
                    help="Multiple curvatures per relation")
parser.add_argument("--rank", default=200, type=int,
                    help="Embedding dimension")
parser.add_argument("--multi-step", action='store_true',
                    help="do multi-steps inference without ground truth")
parser.add_argument("--topk", type=int, default=0,
                    help="choose top k entities as results when do multi-steps without ground truth")
parser.add_argument("--encoder", choices=['gc'], default='gc',
                    help="method of encoder")
parser.add_argument("--dropout", "--en-dropout", type=float, default=0.,
                    help="dropout probability")
parser.add_argument("--n-layers", type=int, default=2,
                    help="number of propagation rounds")
parser.add_argument("--en-loop", action='store_true', default=True,
                    help="mask in gc")
parser.add_argument('--en-bias', action='store_true', default=True,
                    help='')
parser.add_argument("--up-dropout", type=float, default=0.,
                    help="dropout probability for updater")
parser.add_argument("--up-delta", action='store_true')
parser.add_argument("--use-time", choices=['', 'add', 'cat'], default='',
                    help="time encoding")
parser.add_argument("--n-head", type=int, default=1,
                    help="multi-head attention")
parser.add_argument("--layer-norm", action="store_true",
                    help="in att updater")
parser.add_argument("--decoder", type=str, default='',
                    help="method of reasoner")
parser.add_argument("--bias", default="constant", type=str, choices=["constant", "learn", "none"],
                    help="Bias type (none for no bias)")
parser.add_argument("--gamma", default=0, type=float,
                    help="Margin for distance-based losses")
parser.add_argument("--de-dropout", type=float, default=0.,
                    help="dropout probability for decoder")
parser.add_argument("--s-hp", type=float, default=-1,
                    help='if >-1, comb according to hp')
parser.add_argument("--s-comb", choices=['mean', 'max', 'min', 'unary'], default='max',
                    help='Two: method to combine 2 scores')
parser.add_argument("--s-softmax", action='store_true')
parser.add_argument("--s-dropout", type=float, default=0.)
parser.add_argument("--s-delta-ind", action='store_true',
                    help="l_delta != r_delta")
parser.add_argument("--reason-dropout", type=float, default=0.)
parser.add_argument("--save-model", type=str, default='', help="save model name")
parser.add_argument("--save-epoch", type=int, default=-1, help="save model epoch")
parser.add_argument("--test", action='store_true', default=False,
                    help="load stat from dir and directly test")

parser = parser.parse_args()

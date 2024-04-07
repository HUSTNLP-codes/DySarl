import json
import logging
import datetime
import torch.optim
import math
from utils import *
import os.path
from model import MGCN
from optimizer import *
import sys
from config import parser
import torch.nn as nn

def set_logger(args, logName):
    save_dir = get_savedir(args.dataset)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, logName)
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    print("Saving logs in: {}".format(save_dir))
    return save_dir


def train(args):
    torch.manual_seed(2024)

    save_dir = set_logger(args, "train.log")
    with open(os.path.join(save_dir, "config.json"), 'a') as fjson:
        json.dump(vars(args), fjson)

    model_name = "model_d{}-ly{}-dp{}".format(args.rank, args.n_layers, args.dropout)
    model_path = os.path.join(save_dir, '{}'.format(model_name))
    logging.info("Dimension = {}".format(args.rank))
    logging.info("Layer = {}".format(args.n_layers))
    logging.info("lr = {}".format(args.learning_rate))
    logging.info("Dropout = {}".format(args.dropout))
    logging.info("Neg sampling size = {}".format(args.neg_sample_size))
    logging.info(args)

    if args.double_precision:
        torch.set_default_dtype(torch.float64)
    else:
        torch.set_default_dtype(torch.float32)

    data_path = "../data"
    dataset = Dataset(data_path, args.dataset)
    args.sizes = dataset.get_shape()
    logging.info("\t " + str(dataset.get_shape()))
    train_data = dataset.get_train()
    valid_data = dataset.get_valid()
    test_data = dataset.get_test()

    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        ValueError("WARNING: CUDA is not available!!!")
    args.device = torch.device("cuda:0" if use_cuda else "cpu")

    model = MGCN(args)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    model.to(args.device)
    model.cuda(args.device)

    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    optimizer = KGOptimizer(model, optim_method, args.dataset, args.valid_freq, args.multi_step, args.topk,
                            args.batch_size, args.neg_sample_size, bool(args.double_neg), use_cuda, args.dropout)

    if args.test:
        logging.info("\t ---------------------------Start Testing!---------------------------")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logging.info("Evaluation Test Set:")
        _, rank, filter_rank = optimizer.evaluate(test_data, valid_mode=False, epoch=-1)
        test_metrics_raw = compute_metrics(rank)
        test_metrics_filter = compute_metrics(filter_rank)
        logging.info(format_metrics(test_metrics_raw, split="Raw test"))
        logging.info(format_metrics(test_metrics_filter, split="Filtered test"))
    else:
        counter = 0
        best_mrr = None
        best_epoch = None
        logging.info("\t ---------------------------Start Training!-------------------------------")
        for epoch in range(args.max_epochs):
            model.train()
            if use_cuda:
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            loss = optimizer.epoch(train_data, epoch=epoch)
            logging.info("\t Epoch {} | average train loss: {:.4f}".format(epoch, loss))
            if math.isnan(loss.item()):
                break
            model.eval()
            valid_loss, ranks, filter_ranks = optimizer.evaluate(valid_data, valid_mode=True, epoch=epoch)
            logging.info("\t Epoch {} | average valid loss: {:.4f}".format(epoch, valid_loss))

            if (epoch + 1) % args.valid_freq == 0:
                valid_metrics = compute_metrics(ranks)
                logging.info(format_metrics(valid_metrics, split="Raw valid"))
                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = epoch
                    logging.info("\t Saving model at epoch {} in {}".format(epoch, save_dir))
                    torch.save(model.cpu().state_dict(), model_path)
                    if use_cuda:
                        model.cuda()
                else:
                    counter += 1
                    if counter == args.patience:
                        logging.info("\t Early stopping")
                        break
                    elif counter == args.patience // 2:
                        pass
        logging.info("\t ---------------------------Optimization Finished!---------------------------")
        if best_mrr:
            logging.info("\t Saving best model at epoch {}".format(best_epoch))
    return None

if __name__ == "__main__":
    print("main.py")
    start = datetime.datetime.now()
    train(parser)
    end = datetime.datetime.now()
    logging.info('total runtime: %s' % str(end - start))
    sys.exit()

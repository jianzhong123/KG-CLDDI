import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="KGCL")
    parser.add_argument('--bpr_batch', type=int,default=256)#2048
    parser.add_argument('--recdim', type=int,default=512)#128
    parser.add_argument('--layer', type=int,default=3)
    parser.add_argument('--lr', type=float,default=1e-3)
    parser.add_argument('--decay', type=float,default=1e-4)
    parser.add_argument('--dropout', type=int,default=1)#1
    parser.add_argument('--keepprob', type=float,default=0.7)
    parser.add_argument('--a_fold', type=int,default=100)
    parser.add_argument('--testbatch', type=int,default=4096)#4096
    parser.add_argument('--dataset', type=str,default='Drugbank')
    parser.add_argument('--path', type=str,default="./checkpoints")
    parser.add_argument('--topks', nargs='?',default="[10]")
    parser.add_argument('--tensorboard', type=int, default=0)
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--model', type=str, default='kgc')
    parser.add_argument('--test_file', type=str, default='test.txt')
    parser.add_argument('--save', type=int, default=1)
    return parser.parse_args()

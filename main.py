import argparse
import time
import sys
sys.path.append('example')
sys.path.append('src')
from utils import mkdir
from sentimentTestSuite import sentimentTrainModel, sentimentGenerateTestSuite
from mnistTestSuite import mnist_lstm_train, mnist_lstm_test
from record import record


def main():
    
    parser = argparse.ArgumentParser(description='testing for recurrent neural networks')
    parser.add_argument('--model', dest='modelName', default='sentiment', help='')
    parser.add_argument('--TestCaseNum', dest='TestCaseNum', default='200', help='')
    parser.add_argument('--threshold_CC', dest='threshold_CC', default='6', help='')
    parser.add_argument('--threshold_MC', dest='threshold_MC', default='0.7', help='')
    parser.add_argument('--symbols_SQ', dest='symbols_SQ', default='3', help='')
    parser.add_argument('--mode', dest='mode', default='test', help='')
    parser.add_argument('--output', dest='filename', default='./log_folder/record.txt', help='')
    
    args=parser.parse_args()
    
    modelName = args.modelName
    mode = args.mode
    filename = args.filename
    threshold_CC = args.threshold_CC
    threshold_MC = args.threshold_MC
    symbols_SQ = args.symbols_SQ
    TestCaseNum = args.TestCaseNum
    # reset output folder
    mkdir("adv_output")
    mkdir("output")
    # record time
    r = record(filename,time.time())

    if modelName == 'sentiment': 
        if mode == 'train': 
            sentimentTrainModel()
        else: 
            sentimentGenerateTestSuite(r,threshold_CC,threshold_MC,symbols_SQ,TestCaseNum)

    elif modelName == 'mnist':
        if mode == 'train':
            mnist_lstm_train()
        else:
            mnist_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,TestCaseNum)
        
    else: 
        print("Please specify a model from {sentiment, mnist}")
    
    r.close()

if __name__ == "__main__":

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
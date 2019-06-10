import keras
from keras.datasets import imdb 
from keras.layers import *
from keras import *
from keras.models import *
import copy
import keras.backend as K
from keras.preprocessing import sequence 
import itertools as iter
from sentimentClass import Sentiment
from testCaseGeneration import *
from utils import lp_norm, powerset
from testObjective import *
from oracle import *
from record import writeInfo
import random
from scipy import io
from eda import *

def sentimentTrainModel(): 

    sm = Sentiment()
    sm.train_model()

def sentimentGenerateTestSuite(r,threshold_CC,threshold_MC,symbols_SQ,TestCaseNum,minimalTest,TargMetri,CoverageStop):
    r.resetTime()
    random.seed(1)
    # set oracle radius
    oracleRadius = 0.2
    # load model
    sm = Sentiment()
    sm.load_model()
    # test layer
    layer = 1

    # minimal test dataset generation
    if minimalTest != '0':
        ncdata = []
        ccdata = []
        mcdata = []
        sqpdata = []
        sqndata = []

    # predict sentiment from reviews
    review = "i really dislike the movie"
    tmp = sm.fromTextToID(review)
    test = np.squeeze(sm.pre_processing_x(tmp))
    h_t, c_t, f_t = sm.cal_hidden_state(test)

    # input seeds
    X_train = sm.X_train[random.sample(range(20000),5000)]

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    nctoe.model = sm.model
    nctoe.testObjective.layer = layer
    nctoe.testCase = test
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()

    # test objective CC
    cctoe = CCTestObjectiveEvaluation(r)
    cctoe.model = sm.model
    cctoe.testObjective.layer = layer
    cctoe.hidden = h_t
    cctoe.threshold = float(threshold_CC)
    activations_cc = cctoe.get_activations()
    total_features_cc = (np.argwhere(activations_cc >= np.min(activations_cc))).tolist()
    cctoe.testObjective.feature = total_features_cc
    cctoe.testObjective.setOriginalNumOfFeature()
    cctoe.testObjective.setfeaturecount()

    # test objective MC
    mctoe = MCTestObjectiveEvaluation(r)
    mctoe.model = sm.model
    mctoe.testObjective.layer = layer
    mctoe.hidden = f_t
    mctoe.threshold = float(threshold_MC)
    activations_mc = mctoe.get_activations()
    total_features_mc = (np.argwhere(activations_mc >= np.min(activations_mc))).tolist()
    mctoe.testObjective.feature = total_features_mc
    mctoe.testObjective.setOriginalNumOfFeature()
    mctoe.testObjective.setfeaturecount()

    # test objective SQ
    sqtoe = SQTestObjectiveEvaluation(r)
    sqtoe.model = sm.model
    sqtoe.testObjective.layer = layer
    sqtoe.symbols = int(symbols_SQ)
    # generate all the features
    # choose time steps to cover
    indices = slice(480, 485)
    # characters to represent time series
    alpha_list = [chr(i) for i in range(97, 97 + int(symbols_SQ))]
    symb = ''.join(alpha_list)
    sqtoe.testObjective.feature_p = list(iter.product(symb, repeat=5))
    sqtoe.testObjective.feature_n = list(iter.product(symb, repeat=5))
    sqtoe.testObjective.setOriginalNumOfFeature()


    for test in X_train:
        for i in range(4):
            text = sm.fromIDToText(test)
            (label1, conf1) = sm.displayInfo(test)
            # get next input test2
            # test case pertubations
            alpha = random.uniform(0.001, oracleRadius)
            aug_text = eda(text, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=1)
            tmp = sm.fromTextToID(str(aug_text[0]))
            test2 = np.squeeze(sm.pre_processing_x(tmp))

            if not (test2 is None):
                (label2, conf2) = sm.displayInfo(test2)
                h_t, c_t, f_t = sm.cal_hidden_state(test2)
                cctoe.hidden = h_t
                sm.updateSample(label2, label1, alpha, True)
                # update NC coverage
                nctoe.testCase = test2
                nctoe.update_features()
                # update CC coverage
                cctoe.hidden = h_t
                cctoe.update_features()
                # update MC coverage
                mctoe.hidden = f_t
                mctoe.update_features()
                # update SQ coverage
                sqtoe.hidden = h_t
                sqtoe.update_features(indices)
                # write information to file
                writeInfo(r, sm.numSamples, sm.numAdv, sm.perturbations,nctoe.coverage,cctoe.coverage, mctoe.coverage, sqtoe.coverage_p,sqtoe.coverage_n)

                # terminate condition
                if TargMetri == 'CC':
                    termin = cctoe.coverage
                elif TargMetri == 'GC':
                    termin = mctoe.coverage
                elif TargMetri == 'SQN':
                    termin = sqtoe.coverage_n
                elif TargMetri == 'SQP':
                    termin = sqtoe.coverage_p
                else:
                    termin = 0

                # output test cases and adversarial examples
                if minimalTest == '0':
                    f = open('output/test_set.txt', 'a')
                    f.write(str(label1))
                    f.write('\t')
                    f.writelines(str(aug_text[0]))
                    f.write('\n')
                    f.close()
                    if label2 != label1 :
                        f = open('adv_output/adv_test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.write(str(label2))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()

                else:
                    if nctoe.minimal == 1 :
                        ncdata.append(test2)
                        f = open('minimal_nc/test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()
                    if cctoe.minimal == 1 :
                        ccdata.append(test2)
                        f = open('minimal_cc/test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()
                    if mctoe.minimal == 1 :
                        mcdata.append(test2)
                        f = open('minimal_mc/test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()
                    if sqtoe.minimalp == 1 :
                        sqpdata.append(test2)
                        f = open('minimal_sqp/test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()
                    if sqtoe.minimaln == 1 :
                        sqndata.append(test2)
                        f = open('minimal_sqn/test_set.txt', 'a')
                        f.write(str(label1))
                        f.write('\t')
                        f.writelines(str(aug_text[0]))
                        f.write('\n')
                        f.close()

            # check termination condition
            if sm.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
                continue
            else:
                io.savemat('log_folder/feature_count_CC.mat', {'feature_count_CC': cctoe.testObjective.feature_count})
                io.savemat('log_folder/feature_count_MC.mat', {'feature_count_MC': mctoe.testObjective.feature_count})
                # if minimalTest != '0':
                #     np.save('minimal_nc/ncdata', ncdata)
                #     np.save('minimal_cc/ccdata', ccdata)
                #     np.save('minimal_mc/mcdata', mcdata)
                #     np.save('minimal_sqp/sqpdata', sqpdata)
                #     np.save('minimal_sqn/sqndata', sqndata)
                break
        if sm.numSamples < int(TestCaseNum) and termin < float(CoverageStop):
            continue
        else:
            break

    print("statistics: \n")
    nctoe.displayCoverage()
    cctoe.displayCoverage()
    mctoe.displayCoverage()
    sqtoe.displayCoverage1()
    sqtoe.displayCoverage2()
    sm.displaySamples()
    sm.displaySuccessRate()


    
    

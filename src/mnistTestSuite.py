from keras.layers import *
from mnistClass import mnistclass
from keras.preprocessing import image
from scipy import io
import itertools as iter
from testCaseGeneration import *
from testObjective import *
from oracle import *
from record import writeInfo
import random

def mnist_lstm_train():
    mn = mnistclass()
    mn.train_model()

def mnist_lstm_test(r,threshold_CC,threshold_MC,symbols_SQ,TestCaseNum):
    r.resetTime()
    # epsilon value range (a, b]
    random.seed(1)
    a = 0.05
    b = 0.1
    step_bound = 5
    # set up oracle radius
    oracleRadius = 0.005
    # load model
    mn = mnistclass()
    mn.load_model()
    # test layer
    layer = 1

    # test case
    test = mn.X_test[15]
    h_t, c_t, f_t = mn.cal_hidden_state(test, layer)

    # input seeds
    X_train = mn.X_train[random.sample(range(20000), 5000)]

    # test objective NC
    nctoe = NCTestObjectiveEvaluation(r)
    nctoe.model = mn.model
    nctoe.testObjective.layer = layer
    nctoe.testCase = test
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()

    # test objective CC
    cctoe = CCTestObjectiveEvaluation(r)
    cctoe.model = mn.model
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
    mctoe.model = mn.model
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
    sqtoe.model = mn.model
    sqtoe.testObjective.layer = layer
    sqtoe.symbols = int(symbols_SQ)
    # generate all the features
    # choose time steps to cover
    indices = slice(19, 24)
    # characters to represent time series
    alpha_list = [chr(i) for i in range(97,97+int(symbols_SQ))]
    symb = ''.join(alpha_list)
    sqtoe.testObjective.feature_p = list(iter.product(symb, repeat=5))
    sqtoe.testObjective.feature_n = list(iter.product(symb, repeat=5))
    sqtoe.testObjective.setOriginalNumOfFeature()

    # get gradient function for the mnist
    f, nodes_names = get_gradients_function(mn.model, mn.layerName(0))

    for test in X_train:
        for i in range(4):
            o = oracle(test, 2, oracleRadius)
            last_activation = np.squeeze(mn.model.predict(test[np.newaxis, :]))
            (label1, conf1) = mn.displayInfo(test)
            epsilon = random.uniform(a,b)
            # get next input test2 from the current input test
            step = random.randint(1,step_bound)
            test2 = getNextInputByGradient(f, nodes_names, mn, epsilon, test, last_activation,step)

            if not (test2 is None):
                (label2, conf2) = mn.displayInfo(test2)
                h_t, c_t, f_t = mn.cal_hidden_state(test2, layer)
                mn.updateSample(label2,label1,o.measure(test2),o.passOracle(test2))
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
                writeInfo(r,mn.numSamples,mn.numAdv,mn.perturbations,nctoe.coverage,cctoe.coverage,mctoe.coverage,sqtoe.coverage_p,sqtoe.coverage_n)
                # output test cases and adversarial example
                img = test2.reshape((28, 28, 1))
                pred_img = image.array_to_img(img)
                pred_img.save('output/output_%d_%d_%d.jpg' % (mn.numSamples, label1, label2))
                if label2 != label1 and o.passOracle(test2) == True:
                    pred_img.save('adv_output/output_%d_%d_%d.jpg' % (mn.numSamples, label1, label2))

        # check termination condition
        if mn.numSamples >= int(TestCaseNum):
            io.savemat('log_folder/feature_count_CC.mat', {'feature_count_CC': cctoe.testObjective.feature_count})
            io.savemat('log_folder/feature_count_MC.mat', {'feature_count_MC': mctoe.testObjective.feature_count})
            break
    print("statistics: \n")
    nctoe.displayCoverage()
    cctoe.displayCoverage()
    mctoe.displayCoverage()
    sqtoe.displayCoverage1()
    sqtoe.displayCoverage2()
    mn.displaySamples()
    mn.displaySuccessRate()

# Coverage-Guided Testing of Long Short-Term Memory (LSTM) Networks: 

<center>
<img src="img/mnist_sm_adv.png" data-canonical-src="img/mnist_sm_adv.png" width="800" height="600" />
</center>

## Test Metrics and LSTM Models
       
#### Four Test metrics are used: 
1. Neuron Coverage (NC), 
2. Cell Coverage (SC), 
3. Gate Coverage (BC), 
4. Sequence Coverage (TC)

#### Three models trained by LSTM: 
1. Sentiment Analysis, 
2. MNIST Handwritten Digits, 
3. Lipophilicity Prediction (Physical Chemistry)

## Software Dependencies: 

1. rdkit (https://www.rdkit.org/docs/Install.html), by running the following commands: 

       conda create -c rdkit -n my-rdkit-env rdkit
       
       conda activate my-rdkit-env
       
Note: with the above commands, we create a new virtual environment dedicated for rdkit. Below, every time one needs to run the program, he/she needs to activate the my-rdkit-env
      
2. Other packages including 

       conda install -c menpo opencv keras nltk matplotlib
      
       pip install saxpy sklearn

## Command to Run: 

We have two commands to run testing procedure and to run result analysis procedure, respectively. 

#### to run testing procedure

    python main.py --model <modelName> 
                   --TestCaseNum <Num. of Test Cases> 
                   --TargMetri <Terminate Metrics> 
                   --CoverageStop <Terminate Coverage Rate> 
                   --threshold_SC <SC threshold> 
                   --threshold_BC <BC threshold> 
                   --symbols_TC <Num. of symbols> 
                   --seq <seq in cells to test>
                   [--mode <modeName>] 
                   --minimalTest <if generate minimal test set> 
                   --output <output file path>

where 
1. \<modelName> can be in {sentiment,mnist,lipo}
2. \<Num. of Test Cases> is expected number of test cases
3. \<Termination Metrics> can be in {SC,BC,TCN,TCP}
4. \<Termination Coverage Rate> is expected coverage rate of Termination Metric
5. \<SC threshold> can be in [3,9]  
6. \<BC threshold> can be in [0,1]
7. \<Num. of symbols> can be in {2,3}
8. \<seq in cells to test> can be in {mnist:[19,23],sentiment:[480:484],lipo:[70,74]}
8. \<modeName> can be in {train,test} with default value test 
9. \<Generate minimal test set> can be in {0: No, 1: Yes}
10. \<output file path> specifies the path to the output file

For example, we can run the following 

    python main.py --model mnist --TestCaseNum 2000 --TargMetri SC --CoverageStop 0.9 --threshold_SC 6 --threshold_BC 0.8 --symbols_TC 2 --seq [19,23] --minimalTest 0 --output log_folder/record.txt

which says that, we are working with MNIST model, and the test case generation will terminate when either the number of test cases is over 2000 or the target metric, Cell Coverage, reaches 0.9 coverage rate. We need to specify other parameters including threshold_SC, threshold_BC, symbols_TC, seq. Moreover, we do not ask for the generation of miminal test suite, and the log is generated to the file log_folder/record.txt. 

#### to run result analysis procedure
readfile.py can read the log file log_folder\record.txt and .MAT file of test conditions counting. Several figures includes the coverage updating information for all metrics and test conditions statistics plot are generated. 

    python readfile.py --output log_folder/record.txt --metrcis all
    
## Reference 


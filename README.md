# Coverage-Guided Testing of Recurrent Neural Networks: 

#### Four Test metrics are used: 
1. Neuron Coverage (NC), 
2. Cell Coverage (CC), 
3. Gate Coverage (GC), 
4. Sequence Coverage (SQ)

#### Three models trained by LSTM: 
1. Sentiment Analysis, 
2. MNIST Handwritten Digits, 
3. Lipophilicity Prediction (Physical Chemistry)

## Software Dependencies: 

1. rdkit (https://www.rdkit.org/docs/Install.html), by running the following command: 

       conda install -c conda-forge rdkit
      
      

## Command to Run: 

We have two commands to run testing procedure and to run result analysis procedure, respectively. 

#### to run testing procedure

    python main.py --model <modelName> --TestCaseNum <Num. of Test Cases> 
    --TargMetri <Terminate Metrics> --CoverageStop <Terminate Coverage Rate> 
    --threshold_CC <CC threshold> --threshold_GC <GC threshold> 
    --symbols_SQ <Num. of symbols> --mode <modeName> 
    --minimalTest <if generate minimal test set> -- output <output file path>

where 
1. \<modelName> can be in {sentiment,mnist,lipo}
2. \<Num. of Test Cases> is expected number of test cases
3. \<Termination Metrics> can be in {CC,GC,SQN,SQP}
4. \<Termination Coverage Rate> is expected coverage rate of Termination Metric
5. \<CC threshold> can be in [3,9]  
6. \<MC threshold> can be in [0,1]
7. \<Num. of symbols> can be in {2,3}
8. \<Generate minimal test set> can be in {0: No, 1: Yes}

For example, we can run the following 

    python main.py --model mnist --TestCaseNum 2000 --TargMetri CC --CoverageStop 0.9 --threshold_CC 6 --threshold_GC 0.8 --symbols_SQ 2 --minimalTest 0 --output log_folder/record.txt

which says that, both the number of test cases is over 2000 or target metrics, Cell Coverage, reach up to 0.9 coverage rate will induce the termination of the program.

#### run result analysis procedure
readfile.py can read the log file log_folder\record.txt and generate .MAT file containg coverage updating information which can be further plotted with Matlab.

    python readfile.py

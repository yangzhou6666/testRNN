

Testing Recurrent Neural Networks: 

Four Test metrics are used: Neuron Coverage (NC), Cell Coverage (CC), Gate Coverage (GC), Sequence Coverage (SQ)

Three models trained by LSTM: Sentiment Analysis, Mnist Handwritten Digits, Lipophilicity Prediction (Physical
Chemistry)

Command to run: 

> python main.py --model \<modelName> --TestCaseNum \<Num. of Test Cases> --TargMetri \<Terminate Metrics> --CoverageStop \<Terminate Coverage Rate> --threshold_CC \<CC threshold> --threshold_GC \<GC threshold> --symbols_SQ \<Num. of symbols> --mode \<modeName> --minimalTest \<if generate minimal test set> -- output \<output file path>

where 

\<modelName> can be in {sentiment,mnist,lipo}

\<Num. of Test Cases> is expected number of test cases

\<Termination Metrics> can be in {CC,GC,SQN,SQP}

\<Termination Coverage Rate> is expected coverage rate of Termination Metric
  
\<CC threshold> can be in [3,9]  

\<MC threshold> can be in [0,1]

\<Num. of symbols> can be in {2,3}

\<Generate minimal test set> can be in {0: No, 1: Yes}

For example: 

> python main.py --model mnist --TestCaseNum 2000 --TargMetri CC --CoverageStop 0.9 --threshold_CC 6 --threshold_GC 0.8 --symbols_SQ 2 --minimalTest 0 --output log_folder/record.txt

readfile.py can read the log file log_folder\record.txt and generate .MAT file containg coverage updating information which can be further plotted on Matlab.

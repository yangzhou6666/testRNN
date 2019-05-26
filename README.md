

Testing Recurrent Neural Networks: 

Four Test metrics are used: Neuron Coverage (NC), Cell Coverage (CC), Gate Coverage (MC), Sequence Coverage (SQ)

Three models trained by LSTM: Sentiment Analysis, Mnist Handwritten Digits, Lipophilicity Prediction (Physical
Chemistry)

Command to run: 

> python main.py --model \<modelName> --TestCaseNum \<Num. of Test Cases> --threshold_CC \<CC threshold> --threshold_MC \<MC threshold> --symbols_SQ \<Num. of symbols> --mode \<modeName> -- output \<output file path>

where 

\<modelName> can be in {sentiment,mnist,lipo}
  
\<CC threshold> can be in [3,9]  

\<MC threshold> can be in [0,1]

\<Num. of symbols> can be in {2,3}

For example: 

> python main.py --model sentiment --threshold_CC 6 --threshold_MC 0.8 --symbols_SQ 2 --output log_folder/sentimentNC.txt

#!/bin/bash

SCRIPTDIR=$(dirname "$0")

chunkSize=200
jobsNumber=1
modelFile="$SCRIPTDIR/Senti4SD"

help(){
    echo "Usage: classificationTask.sh -i input.csv [-d documents] [-m model] [-c chunk_size] [-j jobs_number] [-o Senti4SD.model]"
    echo "-i train and test data for model training and evaluation. If only train is passed the script split the data in 70% train and 30% test [required]"
    echo "-c chunk size [optional] [defaul = 200]"
    echo "-j number of jobs for parallelism. In case of '-1' value it will use all available cores. [default = -1]"
    echo "-o output file with trained model [default = Senti4SD.model]"
    exit 1
}

NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
    help
    exit 1
fi

while getopts ":h:i:d:m:c:j:o:" OPTIONS; do
    case $OPTIONS in
        h)
          help
          ;;
        i)
          inputFiles+=($OPTARG)
          ;;
        c)
          chunkSize=$OPTARG
          ;;
        j)
          jobsNumber=$OPTARG
          ;;
        m)
          modelFile="$SCRIPTDIR/$OPTARG"
          ;;
        \?)
          echo -e \\n"Option $OPTARG not allowed."
          help
          ;;
    esac
done

INPUTFILESLENGTH=${#inputFiles[@]} 
echo $INPUTFILESLENGTH

if [ $INPUTFILESLENGTH -lt 1 ]; then
    echo "Train data file is required!"
    exit 1
else
if [ $INPUTFILESLENGTH -gt 2 ]; then
    echo "Too many input file!"
    exit 1
else
if [ $INPUTFILESLENGTH -eq 1 ]; then

  inputFile=$inputFiles
  echo "test"
  echo $inputFile

  python $SCRIPTDIR/csv_processing.py -i $inputFile -c text -c polarity

  IFS='.' read -ra FILENAMESPLIT <<< "$inputFile"
  jarInputFile="${FILENAMESPLIT[0]}_jar.csv"

  echo $jarInputFile

    #-F A: all features to be considered
    #-i file_name: a file containg a document for every line
    #-W cbow600.bin: DSM to be loaded
    #-oc file_name.csv: output dataset containg the features extracted
    #-vd numeric: vectors size (for cbow600.bin the size is 600)
    #-L: if present corpus have a label column [optional]
    #-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    #-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

  java -jar $SCRIPTDIR/Senti4SD-fast.jar -F A -i $jarInputFile -W $SCRIPTDIR/dsm.bin -oc $SCRIPTDIR/extractedFeatures.csv -vd 600 -L

  python $SCRIPTDIR/train.py -i $SCRIPTDIR/extractedFeatures.csv -c $chunkSize -j $jobsNumber -m $modelFile
    
  rm $SCRIPTDIR/extractedFeatures.csv
  rm $jarInputFile
else

  for file in ${inputFiles[@]}; do
    if [ ! -f $file ]; then
      echo "File $file not found!"
      exit 1
    fi
  done

  trainFile=${inputFiles[0]}
  testFile=${inputFiles[1]} 

  python $SCRIPTDIR/csv_processing.py -i $trainFile -c text -c polarity
  python $SCRIPTDIR/csv_processing.py -i $testFile -c text -c polarity

  IFS='.' read -ra FILENAMESPLIT <<< "$trainFile"
  jarTrainFile="${FILENAMESPLIT[0]}_jar.csv"

  IFS='.' read -ra FILENAMESPLIT <<< "$testFile"
  jarTestFile="${FILENAMESPLIT[0]}_jar.csv"

  echo $jarTrainFile
  echo $jarTestFile

    #-F A: all features to be considered
    #-i file_name: a file containg a document for every line
    #-W cbow600.bin: DSM to be loaded
    #-oc file_name.csv: output dataset containg the features extracted
    #-vd numeric: vectors size (for cbow600.bin the size is 600)
    #-L: if present corpus have a label column [optional]
    #-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    #-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

  java -jar $SCRIPTDIR/Senti4SD-fast.jar -F A -i $jarTrainFile -W $SCRIPTDIR/dsm.bin -oc $SCRIPTDIR/extractedFeaturesTrain.csv -vd 600 -L
  java -jar $SCRIPTDIR/Senti4SD-fast.jar -F A -i $jarTestFile -W $SCRIPTDIR/dsm.bin -oc $SCRIPTDIR/extractedFeaturesTest.csv -vd 600 -L

  python $SCRIPTDIR/train.py -i $SCRIPTDIR/extractedFeaturesTrain.csv -i $SCRIPTDIR/extractedFeaturesTest.csv -c $chunkSize -j $jobsNumber -m $modelFile
    
  rm $SCRIPTDIR/extractedFeaturesTrain.csv
  rm $SCRIPTDIR/extractedFeaturesTest.csv
  rm $jarTrainFile
  rm $jarTestFile

fi
fi
fi

#!/bin/bash

SCRIPTDIR=$(dirname "$0")

csvDelimiter='c'
chunkSize=1000
jobsNumber=1
modelFile="$SCRIPTDIR/Senti4SD"

help(){
    echo "Usage: classificationTask.sh -i input.csv [-d documents] [-m model] [-c chunk_size] [-j jobs_number] [-o Senti4SD.model]"
    echo "-i train and test data for model training and evaluation. If only train is passed the script split the data in 70% train and 30% test [required]"
    echo '-d delimiter used in csv file, "c" for comma or "sc" for semicolon'
    echo "-c chunk size [optional] [default = 1000]"
    echo "-j number of jobs for parallelism. In case of '-1' value it will use all available cores. [optional] [default = -1]"
    echo "-o output file with trained model [optional][default = Senti4SD.model]"
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
        d)
          csvDelimiter=$OPTARG
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

  mkdir -p $SCRIPTDIR/temp_features;

  inputFile=$inputFiles

  python $SCRIPTDIR/python/csv_processing.py -i $inputFile -d $csvDelimiter -c text -c polarity

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

  python $SCRIPTDIR/python/train.py -i $SCRIPTDIR/temp_features/extractedFeatures.csv -c $chunkSize -j $jobsNumber -m $modelFile
    
  rm -rf $SCRIPTDIR/temp_features 
  rm $jarInputFile
else

  for file in ${inputFiles[@]}; do
    if [ ! -f $file ]; then
      echo "File $file not found!"
      exit 1
    fi
  done

  mkdir -p $SCRIPTDIR/temp_features;

  trainFile=${inputFiles[0]}
  testFile=${inputFiles[1]} 

  python $SCRIPTDIR/python/csv_processing.py -i $trainFile -d $csvDelimiter -c Text -c Polarity
  python $SCRIPTDIR/python/csv_processing.py -i $testFile -d $csvDelimiter -c Text -c Polarity

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

  java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F A -i $jarTrainFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTrain.csv -vd 600 -L
  java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F A -i $jarTestFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTest.csv -vd 600 -L

  python $SCRIPTDIR/python/train.py -i $SCRIPTDIR/temp_features/extractedFeaturesTrain.csv -i $SCRIPTDIR/temp_features/extractedFeaturesTest.csv -c $chunkSize -j $jobsNumber -m $modelFile
    
  rm -rf $SCRIPTDIR/temp_features  
  rm $jarTrainFile
  rm $jarTestFile

fi
fi
fi

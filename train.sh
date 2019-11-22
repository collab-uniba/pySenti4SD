#!/bin/bash

SCRIPTDIR=$(dirname "$0")

csvDelimiter='c'
features='A'
grams=false
chunkSize=1000
jobsNumber=1
modelFile="$SCRIPTDIR/Senti4SD"

help(){
    echo "Usage-1: sh train.sh -i train.csv [-d csv-delimiter] [-F features] [-g] [-c chunk_size] [-j jobs_number] [-o Senti4SD.model]"
    echo "or"
    echo "Usage-2: sh train.sh -i train.csv -i test.csv [-d csv-delimiter] [-g] [-c chunk_size] [-j jobs_number] [-o Senti4SD.model]"
    echo "-i -- the input file, containing the corpus for the training; it's possible to run the script with two separated datasets, one for training and the other for testing [see Usage-2]. [required]"
    echo '-d -- the delimiter used in the csv file, where c stands for comma and sc for semicolon. [Default value: "c"]'
    echo '-F -- all features to be considered. A stands for all, L stands for lexicon fetures, S stands for semantic features and K stands for keyword features. [Default value: A]'
    echo '-g -- enables the extraction of n-grams (i.e,. bigrams and unigrams)'
    echo "-c -- the number of rows to read from the dataset per time, to avoid high memory usage. [Default value: 1000]"
    echo "-j -- the number of cores to use during csv reading phase. If you pass -1 all cores will be used.
		If you pass a number higher than your total core number, the script will use all the cores. [Default value: 1] "
    echo "-o -- the name of trained model. [Default value: 'Senti4SD.model']"
    exit 1
}

NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
    help
    exit 1
fi

while getopts "h:i:d:F:m:c:j:o:g" OPTIONS; do
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
        F)
          features=$OPTARG
          ;;
	    g)
	      grams=true
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

  java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarInputFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeatures.csv -vd 600 -L

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
  
  if [ "$grams" = true ] ; then
	java -jar $SCRIPTDIR/java/NgramsExtraction.jar $jarTrainFile -L
  

    	#-F A: all features to be considered
    	#-i file_name: a file containg a document for every line
    	#-W cbow600.bin: DSM to be loaded
    	#-oc file_name.csv: output dataset containg the features extracted
    	#-vd numeric: vectors size (for cbow600.bin the size is 600)
    	#-L: if present corpus have a label column [optional]
    	#-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    	#-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

  	java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarTrainFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTrain.csv -vd 600 -L -ul $SCRIPTDIR/UnigramsList -bl 	$SCRIPTDIR/BigramsList
  	java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarTestFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTest.csv -vd 600 -L -ul $SCRIPTDIR/UnigramsList -bl 		$SCRIPTDIR/BigramsList
  else
        #-F A: all features to be considered
    	#-i file_name: a file containg a document for every line
    	#-W cbow600.bin: DSM to be loaded
    	#-oc file_name.csv: output dataset containg the features extracted
    	#-vd numeric: vectors size (for cbow600.bin the size is 600)
    	#-L: if present corpus have a label column [optional]
    	#-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    	#-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

  	java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarTrainFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTrain.csv -vd 600 -L
  	java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarTestFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeaturesTest.csv -vd 600 -L
  fi

  python $SCRIPTDIR/python/train.py -i $SCRIPTDIR/temp_features/extractedFeaturesTrain.csv -i $SCRIPTDIR/temp_features/extractedFeaturesTest.csv -c $chunkSize -j $jobsNumber -m $modelFile
    
  rm -rf $SCRIPTDIR/temp_features  
  rm $jarTrainFile
  rm $jarTestFile

fi
fi
fi

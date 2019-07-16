#!/bin/bash

SCRIPTDIR=$(dirname "$0")

inputFile=""
csvDelimiter='c'
features='A'
grams=false
documents=false
model="$SCRIPTDIR/Senti4SD.model"
chunkSize=200
jobsNumber=1
outputFile="$SCRIPTDIR/predictions.csv"

help(){
    echo "Usage: sh classification.sh -i input.csv [-d delimiter] [-F features] [-g] [-t] [-m model] [-c chunk_size] [-j jobs_number] [-o predictions.csv]"
    echo "-i input file to classify [required]"
    echo '-d delimiter used in csv file, "c" for comma or "sc" for semicolon'
    echo '-F -- all features to be considered. A stands for all, L stands for lexicon fetures, S stands for semantic features and K stands for keyword features. [Default value: A]'
    echo "-g -- enables use of custom UnigramsList and BigramsList [optional]"
    echo "-t -- enables documents saving along with the prediction labels inside 'predictions.csv' file. [optional]"
    echo "-m prediction model [default = Senti4SD]"
    echo "-c chunk size [default = 200]"
    echo "-j number of jobs for parallelism. In case of '-1' value it will use all available cores [default = -1]"
    echo "-o output file with predicted label [default = predictions.csv]"
    exit 1
}

NUMARGS=$#
if [ $NUMARGS -eq 0 ]; then
    help
    exit 1
fi

while getopts "h:i:d:F:m:c:j:o:tg" OPTIONS; do
    case $OPTIONS in
        h)
          help
          ;;
        i)
          inputFile=$OPTARG
          ;;
        d)
          csvDelimiter="$OPTARG"
          ;;
        t)
          documents=true
          ;;
	    g)
	      grams=true
          ;;
        F)
          features=$OPTARG
          ;;
        m)
          model="$SCRIPTDIR/$OPTARG"
          ;;
        c)
          chunkSize=$OPTARG
          ;;
        j)
          jobsNumber=$OPTARG
          ;;
        o)
          outputFile="$SCRIPTDIR/$OPTARG"
          ;;
        \?)
          echo -e \\n"Option $OPTARG not allowed."
          help
          ;;
    esac
done

if [ -z $inputFile ]; then
    echo "input csv file is required!"
    exit 1
fi
if [ ! -f $inputFile ]; then
    echo "File $inputFile not found!"
    exit 1
fi 

mkdir -p $SCRIPTDIR/temp_features;

python $SCRIPTDIR/python/csv_processing.py -i $inputFile -d $csvDelimiter -c text

IFS='.' read -ra FILENAMESPLIT <<< "$inputFile"
jarInputFile="${FILENAMESPLIT[0]}_jar.csv"

if [ "$grams" = true ] ; then
    unigramsFile="$SCRIPTDIR/UnigramsList"
    bigramsFile="$SCRIPTDIR/BigramsList"
    echo $unigramsFile
    echo $bigramsFile
    if [ ! -f $unigramsFile ]; then
    	echo "File $unigramsFile not found!"
    	exit 1
    fi
    if [ ! -f $bigramsFile ]; then
	    echo "File $bigramsFile not found!"
    	exit 1
    fi

    #-F A: all features to be considered
    #-i file_name: a file containg a document for every line
    #-W cbow600.bin: DSM to be loaded
    #-oc file_name.csv: output dataset containg the features extracted
    #-vd numeric: vectors size (for cbow600.bin the size is 600)
    #-L: if present corpus have a label column [optional]
    #-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    #-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

    java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarInputFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeatures.csv -vd 600 -ul $unigramsFile -bl $bigramsFile

    if [ "$documents" = true ] ; then
        python $SCRIPTDIR/python/classification_task.py -i $SCRIPTDIR/temp_features/extractedFeatures.csv -i $inputFile -d $csvDelimiter -t -m $model -c $chunkSize -j $jobsNumber -o $outputFile
    else
        python $SCRIPTDIR/python/classification_task.py -i $SCRIPTDIR/temp_features/extractedFeatures.csv -i $inputFile -d $csvDelimiter -m $model -c $chunkSize -j $jobsNumber -o $outputFile
    fi
    
    rm -rf $SCRIPTDIR/temp_features
    rm $jarInputFile
else
    #-F A: all features to be considered
    #-i file_name: a file containg a document for every line
    #-W cbow600.bin: DSM to be loaded
    #-oc file_name.csv: output dataset containg the features extracted
    #-vd numeric: vectors size (for cbow600.bin the size is 600)
    #-L: if present corpus have a label column [optional]
    #-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    #-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

    java -jar $SCRIPTDIR/java/Senti4SD-fast.jar -F $features -i $jarInputFile -W $SCRIPTDIR/java/dsm.bin -oc $SCRIPTDIR/temp_features/extractedFeatures.csv -vd 600

    if [ "$documents" = true ] ; then
        python $SCRIPTDIR/python/classification_task.py -i $SCRIPTDIR/temp_features/extractedFeatures.csv -i $inputFile -d $csvDelimiter -t -m $model -c $chunkSize -j $jobsNumber -o $outputFile
    else
        python $SCRIPTDIR/python/classification_task.py -i $SCRIPTDIR/temp_features/extractedFeatures.csv -i $inputFile -d $csvDelimiter -m $model -c $chunkSize -j $jobsNumber -o $outputFile
    fi
    
    rm -rf $SCRIPTDIR/temp_features
    rm $jarInputFile
fi

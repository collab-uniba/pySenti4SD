#!/bin/bash

SCRIPTDIR=$(dirname "$0")

inputFile=""
documents="no"
model="$SCRIPTDIR/Senti4SD_model.clf"
chunkSize=200
jobsNumber=1
outputFile="$SCRIPTDIR/predictions.csv"

help(){
    echo "Usage: classificationTask.sh -i input.csv [-d documents] [-m model] [-c chunk_size] [-j jobs_number] [-o predictions.csv]"
    echo "-i input file to classify [required]"
    echo "-d input file to classify [required]"
    echo "-m trained model [optional] [default = Senti4SD_model]"
    echo "-c chunk size [optional] [defaul = 500]"
    echo "-j number of jobs for parallelism [optional] [default = 1]"
    echo "-o output file with predicted label [optional] [default = predictions.csv]"
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
          inputFile=$OPTARG
          ;;
        d)
          documents="$OPTARG"
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
else
if [ ! -f $inputFile ]; then
    echo "File $inputFile not found!"
    exit 1
else 

    python $SCRIPTDIR/csv_processing.py -i $inputFile

    IFS='.' read -ra FILENAMESPLIT <<< "$inputFile"
    jarInputFile="${FILENAMESPLIT[0]}_jar.csv"

    #-F A: all features to be considered
    #-i file_name: a file containg a document for every line
    #-W cbow600.bin: DSM to be loaded
    #-oc file_name.csv: output dataset containg the features extracted
    #-vd numeric: vectors size (for cbow600.bin the size is 600)
    #-L: if present corpus have a label column [optional]
    #-ul file_name: unigram's list to use for feature extraction. If not present default Senti4SD unigram's list will be used [optional]
    #-bl file_name: bigram's list to use for feature extraction. If not present default Senti4SD bigram's list will be used [optional]

    java -jar $SCRIPTDIR/Senti4SD-fast.jar -F A -i $jarInputFile -W $SCRIPTDIR/dsm.bin -oc $SCRIPTDIR/extractedFeatures.csv -vd 600

    python $SCRIPTDIR/classification_task.py -i $SCRIPTDIR/extractedFeatures.csv -i $inputFile -d $documents -m $model -c $chunkSize -j $jobsNumber -o $outputFile
    
    rm $SCRIPTDIR/extractedFeatures.csv
    rm $jarInputFile

fi
fi

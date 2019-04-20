#!/bin/bash
if [ -n `which java` ]; then
	echo "Java is installed"
	if [ -n `which python` ]; then
                echo "python is installed"
		#this will install python library numpy, pandas and sklearn if not already installed
		pip install -r requirements.txt
        else echo "python is not installed!"
	fi
else echo "Java is not installed!"
fi
#!/usr/bin/env bash

# Get directory name https://stackoverflow.com/a/4774063
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

rm -rfv $SCRIPTPATH/results
rm -rfv $SCRIPTPATH/data
rm -rfv $SCRIPTPATH/fig

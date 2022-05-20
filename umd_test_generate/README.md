# UMD Test Generation for SAIL-ON Symbiant 
Copyright 2008-2022 by Raytheon BBN Technologies.  All Rights Reserved

This code is part of our BBN Symbiant team's work in DARPA SAIL-ON, 2021-2022.
The University of Maryland had a TA1 team that hosted a SVO "action classification" task,
which was one of the tasks on which our novelty-aware TA2 system was tested.
This package analyzes the training and validation data from UMD and generates
and scores sample tests similar to UMD's official tests.

The UMD task was to predict the subject (S), verb (V), and object (O) types in images.
Bounding boxes for the S and O regions in the image were supplied in many cases.
If both S and O boxes were provided, the V box was the smallest rectangle 
covering them both.
The training data included many known examples where the S and O boxes were both supplied
and where numeric classes were assigned to the S, V, and O values. 
For example, (2, 1, 4) was (dog, catching, frisbee).
There were also cases where the S box was missing, with only the O box specified;
a special -1 "missing" value was the correct answer for S and V in those cases, 
and similarly if the O box was missing.

The validation data (and some training data) included novel cases 
where the S, V, or O were of novel types not seen in training,
tagged with ID 0.
An image could also be tagged as novel if that combination
of known S, V, and O types had never been seen in training,
but the UMD validation data did not include any such examples.
(A small number of novel cases were included in the training
to provide the UMD baseline system with training data for those classes,
but our system did not use those instances for training.)
The S box had to be present for the V to be specified,
and a special "novel/unknown" value was used for the V in some cases
where UMD felt that systems would not have sufficient data
to distinguish between those two answers. 

This package includes:
- a few freestanding corpus analysis scripts in the directory corpus_analysis
- 'process_dataset.py', which collects and classifies examples from the UMD data
- 'gen_tests.py', which generates sets of sample tests from the processed examples

## Corpus Analysis Scripts

The script 'do_anno_anal.py' is the main one here. 
The other two scripts are special-purpose modified versions.
This script reads through the two annotation files supplied by UMD
for their training and validation data
and outputs a text file with useful statistics.
To run it, set the initial variables for the source data directory,
the initial string for the annotation file names 
(the portion before "train" and "val"),
add the output file path.

## Process_dataset.py

The 'process_dataset.py' script processes 
the train and validation annotation files from UMD,
classifies the different instances while printing out a log file
with information about what it's finding, and then saves the results
to a pickle file that can be used later to generate sample tests.

In addition to predicting SVO triples, systems had to predict
whether or not each image was novel. 
Late in Phase 2, UMD realized that combining the unseen and novel values for Vs
left systems with no way of knowing if such images were novel,
so they provided an expanded "master" version of their validation data
that supplied that additional distinction, 
and this code uses that "master" version of their validation data.
Information from the "master" version 
is also needed to support instance class feedback in the UMD API. 

The script first processes the training data, 
recording and classifying each instance,
and separating out the training instances that do include novelty.
It then processes the validation data, 
skipping a few invalid cases where both S and O were tagged as novel,
and bins the valid examples by whether they contain
a novel S, novel V, novel O, or no novelty.

For each example (training or validation), the script generates 
the line of CSV text that will be needed to specify that example
in the list of examples for a generated sample test,
and that data is saved in a pickle file
for use by the 'gen_tests.py' script.

## Gen_tests.py

The 'gen_tests.py' script generates a suite of tests,
based on the parameters supplied in a test configuration file
and using the pickle file from running 'process_dataset.py'
The suite of tests are output in a directory tree structure
in the format used by UMD's test API.

Each test suite includes tests of four different types:
 - no novelty
 - novel S
 - novel V
 - novel O

Each test has a length, the count of images,
and a round_size, the number of images that the API
will present in each batch.
Tests with novelty also have a 'red_button' point,
at which the source distribution switches to the post-novel one,
and an 'alpha' value, specifying what percentage of the post-red-button
images should be novel. 

The test config file is a two-line CSV file
that specifies parameter values: 
 - batch_num
 - count: number of tests of each type to generate 
 - no_novel_test_len: length of the initial no-novel tests
 - no_novel_round_size: round size for the initial no-novel tests
 - test_len: length of the tests with novelty
 - round_size: round size for the tests with novelty
 - red_button: index value for the switch to the post-novel distribution
 - alpha: percentage of post-red-button images that will be novel

Here is an example config file:
```
batch_num,count,no_novel_test_len,no_novel_round_size,test_len,round_size,red_button,alpha
2,1,600,10,600,10,80,0.3
```

The UMD API supports both novelty and item classification feedback. 
Classification feedback was provided as 5-tuples of class IDs
from a different but related class ontology.
To provide this feedback, the API requires three 
'topk' files. 
UMD provided these in their API repo,
but they are also provided in the 'topk_files' directory here
for convenience.

## Scoring SailOn UMD SVO task runs

Copyright 2021-2022 by Raytheon BBN Technologies.  All Rights Reserved

This component was developed by the BBN/OSU Symbiant team for DARPA SAIL-ON.
To test novelty-capable AI components in SAIL-ON, 
TA1 teams generated tests that included novelty,
and TA2 teams developed AI systems addressing those tasks.
The University of Maryland TA1 team provided an "action classification" task
where the goal was to recognize SVO (subject, verb, object) triples
in images, given bounding boxes identifying the subject and object,
including recognizing novel cases not seen in training.

To run a set of tests, you run your system with the UMD API 
pointed at a directory containing a set of tests.
As you run the system, the API will populate a system output directory.
This scoring script processes that output directory
and generates a directory of score result files.

This scoring script is a short-cut alternative 
to the official evaluation function, which is included in the UMD API.
However, that function requires input files recording the baseline system performance,
which the TA2 teams don't have.
The official Evaluate script also produces a CSV output file that then needs to be further processed
to generate the program metrics using scripts that the TA1 teams 
have not generally shared with the TA2 teams.

The `score.py` script takes three arguments:
- `test_root`: the test directory (parent of the test `OND` directory)
- `sys_output_root`: the system output directory (parent of the output `OND` directory)
- `log_dir`: the directory where scoring results should be written

### Test Directory
The test directory, which might be called `api_tests`,
must contain an `OND` directory (for the protocol),
which in turn must contain a `svo_classification` directory (for the task).

The `svo_classification` directory in turn contains a `test_ids.csv` file
that lists the test ids, one per line. 
For each listed test ID, that directory should also contain two test-specific files:
- `OND.100.000_metadata.json`:  
    Defines the protocol, round size, and feedack budgets.
- `OND.100.100_single_df.csv`:  
    For each image in the test, this file lists the path to the image, 
    the dimensions of the bounding boxes, and the correct SVO answers. 

### System Output Directory
The system output directory will contain some JSON config files
(one for the set of tests and one for each individual test)
which this scoring script does not use.

It will also contain am `OND` directory, which contains 
an `svo_classification` directory.
That directory will contain `detection.csv` and `classification.csv` files
for each test, recording the system's output.
The names of those files begin with a UUID `session_id`, 
followed by the `test_id`. 
The API uses the `session_id` to track runs that span multiple sessions,
but this script assumes that there will be just a single id, and ignores its value.

### Scoring Results Directory
The scoring script writes its outputs to a scoring results directory.
There is a `test_id.log` file with image-by-image results for each test,
and a `summary.log` file that summarizes results.
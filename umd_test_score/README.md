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

There is also a `confusion.pdf` file with confusion matrices.
Two confusion matrices are output for each of S, O, and V,
where the rows display the true classes 
and the columns the classes that the system predicted
in its top-ranked tuple.
The first matrix in each pair is normalized by row,
so showing percentage-wise the relative weights of each system output.
The second matrix shows raw counts for each cell.
(If the same image occurs in more than one test
or multiple times in a given test,
each occurrence is counted in the confusion matrices.)

### Generating Image Symlinks Classified by True and Predicted Classes
If you supply the flag `--save_symlinks` and also supply the argument
`--databset_root` with the path to your local copy of the UMD data,
the scoring results directory will also contain an `images` directory
with a tree of category subdirectories and with leaf directories 
that contain symlinks to each image in the tests run
that has the given true and predicted classes.
(Since the tests can contain duplicates, the image directories
may contain fewer entries than the associated cells
in the confusion matrices.)

The top level categorical split is by the true (annotated) class
with the second level for the predicted class.
For example, `images/S/true_02_dog/pred_04_person/pred_04_horse/image_00741.jpg -> <path>`
is a symlink in the directory for images whose true S value was dog
but where the system's top-ranked triple predicted the S as horse.
The V portion of the tree is split into V-S and V-SO 
to distinguish images for which only an S box was specified
from those where both S and O boxes were provided.
According to the UMD rules, when only the S box is provided, 
the V should always be tagged with id 0 =`novel/unknown`. 
(But note that 11 image in the UMD training data are
acknowledged errors that do not follow this rule.)
from pathlib import Path
import os

data_dir = Path('session/data')
test_data_dir = data_dir / 'tests'
results_dir = data_dir / 'results'
sys_results_dir = data_dir / 'sys_results'
sample_test_ids = [path.name.split('_')[0] for path in sorted(list(test_data_dir.iterdir()))]
trial_size = 2500
round_size = 100
round_count = 25
feedback_max_ids = 10

test_data = None

## Supporting functions
def load_test_data():
    global test_data
    test_data = {}

    for test_id in sample_test_ids:
        image_lines = open(test_data_dir / f'{test_id}_single_df.csv', 'r').read().splitlines()

        ## throw away the header line
        image_lines = image_lines[1:]
        
        ## select the subset of vals that the API should return
        api_lines = []
        for image_line in image_lines:
            vals = image_line.split(',')
            ## We just want the name and then image size and bounding box info
            api_vals = [vals[0]] + vals[10:]
            api_lines.append(','.join(api_vals))

        test_data[test_id] = ['\n'.join(api_lines[i * round_size:(i + 1) * round_size]) for i in range(round_count)]

def clear_results_dirs():
    for file in results_dir.iterdir():
        os.remove(file)
    for file in sys_results_dir.iterdir():
        os.remove(file)

def novel_p(image_path):
    substrings = ['novel_val', 'imgs', 'verb_dataset', 'obj_dataset', 'sub_dataset']
    return any([substring in image_path for substring in substrings])

## API stubs

def test_ids():
    return sample_test_ids

def session_id():
    return '82b41225-f802-4ea0-ab53-648302261475'

def metadata():
    return {'round_size' : round_size,
            'feedback_max_ids' : feedback_max_ids,
            'max_novel_classes' : 5,
            }

def image_data(test_id, round_id):
    global test_data

    if test_data == None:
        load_test_data()
    if round_id >= round_count:
        return None

    return test_data[test_id][int(round_id)]

def detection_feedback(test_id, round_id, image_paths):
    return [(image_path, novel_p(image_path)) for image_path in image_paths]

def record_results(test_id, round_id, detection_file, classification_file):
    open(results_dir / f'{test_id}_{round_id:03d}_detection.csv', 'w').write(detection_file)
    open(results_dir / f'{test_id}_{round_id:03d}_classification.csv', 'w').write(classification_file)

def finish_test(test_id):
    """
    Cancatenate the by-round results into one detection and one classification file for the test
    """
    for file_type in ['detection', 'classification']:
        round_files = sorted(list(results_dir.glob(f'{test_id}*{file_type}.csv')))
        with open(results_dir / f'{test_id}_{file_type}.csv', 'w') as ofile:
            for round_file in round_files:
                with open(round_file, 'r') as infile:
                    ofile.write(infile.read())

                round_file.unlink()

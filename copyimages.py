import csv
import shutil
import os


filepath = '/nfs/stak/users/ullaham/hpc-share/SailON/phase3/sailon-svo/session/tests/api_tests/OND/svo_classification/OND.101.000_single_df.csv'

with open(filepath, newline = '\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[-1] == '1':
            print(row[0], row[-1])
            if not os.path.exists('./dataset_v4/'+ filepath.split('/')[-1][:13]):
                os.makedirs('./dataset_v4/'+ filepath.split('/')[-1][:13])
            shutil.copyfile(row[0], './dataset_v4/'+ filepath.split('/')[-1][:13] + '/' + row[0].split('/')[-1])

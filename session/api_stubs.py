import pathlib
import numpy as np
import pathlib
import json
import pandas as pd


class APIStubs:
    def __init__(self, path_to_metadata_dir, path_to_csv_dir):
        self.path_to_metadata_dir = pathlib.Path(path_to_metadata_dir)
        self.path_to_csv_dir = pathlib.Path(path_to_csv_dir)
    
        self.metadata = {}
        self.csvs = {}


    
        if not self.path_to_metadata_dir.exists():
            raise Exception(f'metadata path {path_to_metadata_dir} was not found.')

        if not self.path_to_csv_dir.exists():
            raise Exception(f'metadata path {path_to_csv_dir} was not found.')
            
        for p in self.path_to_metadata_dir.glob('*'):
            if p.suffix != '.json':
                continue
                
            with open(p, 'rb') as f:
                j = json.load(f)
                
                test_name = p.name.split('_')[0]
                self.metadata[test_name] = j

        for p in self.path_to_csv_dir.glob('*'):
            if p.suffix != '.csv':
                continue
                
            test_name = p.name.split('_')[0]
            self.csvs[test_name] = p

            
        assert set([k for k in self.csvs.keys()]) == set([k for k in self.metadata.keys()]) 
        
        self.cache = {}
        self.cache['last_csv'] = None
        self.cache['df'] = None
        self.cache['csv_data'] = None
        
    def session_id(self):
        return '82b41225-f802-4ea0-ab53-648302261475'

    def test_ids(self):
        return [k for k in self.csvs.keys()]

    def get_metadata(self, test_id):
        if test_id not in self.metadata:
            raise Exception(f'unknown test id {test_id}')
            
        m = self.metadata[test_id]
    
        return {'round_size' : m['round_size'],
                'feedback_max_ids' : m['feedback_max_ids'],
                'red_light' : m['red_light']}
    
    def image_data(self, test_id, round_id):
        if test_id not in self.metadata:
            raise Exception(f'unknown test id {test_id}')
            
        if self.cache['last_csv'] != test_id:
            self._update_cache(test_id)
        
        data = self.cache['csv_data']
            
        if round_id >= len(data):
            return None
            
        return data[round_id]

    def detection_feedback(self, test_id, round_id, image_paths):
        if test_id not in self.metadata:
            raise Exception(f'unknown test id {test_id}')
            
        if self.cache['last_csv'] != test_id:
            self._update_cache(test_id)
            
        df = self.cache['df']
        
        return [(image_path, df[df['new_image_path'] == image_path]['novel'].to_list()[0]) for image_path in image_paths]

    def _update_cache(self, test_id):
        if test_id not in self.metadata:
            raise Exception(f'unknown test id {test_id}')
            
        m = self.metadata[test_id]
        round_size = m['round_size']        
        
        self.cache['last_csv'] = test_id
        self.cache['df'] = pd.read_csv(self.csvs[test_id])

        with open(self.csvs[test_id], 'r') as f:
            image_lines = f.read().splitlines()
            image_lines = image_lines[1:]
            round_count = int(np.ceil(len(image_lines) / round_size))
    
            api_lines = []
            for image_line in image_lines:
                vals = image_line.split(',')
                api_vals = [vals[0]] + vals[10:-1]
                api_lines.append(','.join(api_vals))
        self.cache['csv_data'] = ['\n'.join(api_lines[i * round_size: min((i + 1) * round_size, len(image_lines))]) for i in range(round_count)]
    
    def clear_results_dirs(self):
        pass

    def record_results(self, test_id, round_id, detection_file, classification_file):
        
        pass

    def finish_test(self, test_id):
        pass

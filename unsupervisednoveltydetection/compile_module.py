import torch
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Max Classifier Logits for Anomaly Detection'
    )
    
    # Model persistence parameters
    parser.add_argument(
        '--classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--confidence-calibrator-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--score-contexts-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--known-combinations-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--module-save-file',
        type = str,
        required = True
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    classifier_state_dict = torch.load(args.classifier_load_file)
    confidence_calibrator_state_dict = torch.load(args.confidence_calibrator_load_file)
    score_context_state_dict = torch.load(args.score_contexts_load_file)
    with open(args.known_combinations_load_file, 'rb') as f:
        known_combinations = pickle.load(f)

    module_state_dict = {}
    module_state_dict['classifier'] = classifier_state_dict
    module_state_dict['confidence_calibrator'] = confidence_calibrator_state_dict
    module_state_dict['known_combinations'] = known_combinations

    state_dict = {}
    state_dict['module'] = module_state_dict
    state_dict['subject_score_context'] = score_context_state_dict['subject_score_context']
    state_dict['object_score_context'] = score_context_state_dict['object_score_context']
    state_dict['verb_score_context'] = score_context_state_dict['verb_score_context']

    torch.save(state_dict, args.module_save_file)

if __name__ == '__main__':
    main()

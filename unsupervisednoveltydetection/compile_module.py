import torch
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Max Classifier Logits for Anomaly Detection'
    )
    
    # Model persistence parameters
    parser.add_argument(
        '--detector-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--calibration-logistic-regressions-load-file',
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
    
    detector_state_dict = torch.load(args.detector_load_file)
    calibration_logistic_regressions_state_dict = torch.load(args.calibration_logistic_regressions_load_file)

    state_dict = {}
    state_dict['module'] = detector_state_dict
    state_dict['case_1_logistic_regression'] = calibration_logistic_regressions_state_dict['case_1_logistic_regression']
    state_dict['case_2_logistic_regression'] = calibration_logistic_regressions_state_dict['case_2_logistic_regression']
    state_dict['case_3_logistic_regression'] = calibration_logistic_regressions_state_dict['case_3_logistic_regression']

    torch.save(state_dict, args.module_save_file)

if __name__ == '__main__':
    main()

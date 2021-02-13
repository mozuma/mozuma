import argparse
from io import BytesIO
import requests

import torch
import torchvision.models as m


def replace_malformed_string(s):

    # The state dictionary for Places contains a 'module.' prefix for all weights
    s = str.replace(s, 'module.', '')

    # The state dictionaries for Places and ImageNet contain periods after 'norm'/'conv'
    s = str.replace(s, 'norm.1', 'norm1')
    s = str.replace(s, 'norm.2', 'norm2')
    s = str.replace(s, 'conv.1', 'conv1')
    s = str.replace(s, 'conv.2', 'conv2')
    return s


def download(model_name, url, fname):
    # Download model
    print("Downloading model")
    response = requests.get(url)

    # If the response wasn't 200, return
    if response.status_code != 200:
        print(f"Error: received status code {response.status_code}")
        return

    # Extracting data from response
    print("Extracting data")
    f = BytesIO(response.content)
    state_dict = torch.load(f, map_location=lambda storage, loc: storage)

    # The Places state dictionaries are stored inside the file
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']

    # Clean the state dictionary names for Places and ImageNet weights
    state_dict = {replace_malformed_string(k): v for k, v in state_dict.items()}

    # Check that all keys in the DenseNet model are now in the state dictionary, and vice-versa
    missing_keys = []
    surplus_keys = []

    model = getattr(m, model_name)()
    model_state_dict = model.state_dict()

    # Aparently there are none of the 'num_batches_tracked' values in the state dicts
    for k, _ in model_state_dict.items():
        if k not in state_dict and 'num_batches_tracked' not in k:
            missing_keys.append(k)
    
    for k, _ in state_dict.items():
        if k not in model_state_dict:
            surplus_keys.append(k)

    # Quit if the dictionaries don't match
    if len(missing_keys) > 0 or len(surplus_keys) > 0:
        print('State Dictionaries do not match:')
        print('    Missing Keys:')
        for k in missing_keys:
            print(f'        {k}')
        print('    Surplus Keys:')
        for k in surplus_keys:
            print(f'        {k}')

        return

    # Split into features and the classifier
    features_sd = {k: v for k, v in state_dict.items()
                   if k.startswith('features')}
                
    classifier_sd = {k: v for k, v in state_dict.items()
                     if k.startswith('classifier')}

    fname_features = fname + '_features.pth.tar'
    fname_classifier = fname + '_classifier.pth.tar'

    torch.save(features_sd, fname_features)
    torch.save(classifier_sd, fname_classifier)
    print(f'Files saved successfully')
    return


def main():
    parser = argparse.ArgumentParser(description='Download the state dictionary for a ' + \
        'pre-trained DenseNet model, and save its feature and classifier weights.')

    parser.add_argument('fname', help='the prefix for the file names of the created files')
    parser.add_argument('url', help='the URL of the state dictionary to download')
    parser.add_argument('-m', '--model', default='densenet161', help='the model for the state dict')

    args = parser.parse_args()

    download(args.model, args.url, args.fname)


if __name__ == "__main__":
    main()

# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license.


def postprocess_attr(dataset_attr_labelmap, label_list, conf_list):
    common_attributes = {
        "white",
        "black",
        "blue",
        "green",
        "red",
        "brown",
        "yellow",
        "small",
        "large",
        "silver",
        "wooden",
        "wood",
        "orange",
        "gray",
        "grey",
        "metal",
        "pink",
        "tall",
        "long",
        "dark",
        "purple",
    }
    common_attributes_thresh = 0.1
    attr_alias_dict = {"blonde": "blond"}
    attr_dict = {}
    for label, conf in zip(label_list, conf_list):
        label = dataset_attr_labelmap[label]
        if label in common_attributes and conf < common_attributes_thresh:
            continue
        if label in attr_alias_dict:
            label_target = attr_alias_dict[label]
        else:
            label_target = label
        if label_target in attr_dict:
            attr_dict[label_target] += conf
        else:
            attr_dict[label_target] = conf
    if len(attr_dict) > 0:
        # the most confident one comes the last
        sorted_dic = sorted(attr_dict.items(), key=lambda kv: kv[1])
        return list(zip(*sorted_dic))
    else:
        return [[], []]

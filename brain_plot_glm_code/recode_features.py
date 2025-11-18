# This script is for extracting, recoding, and exporting features from the annotation excel sheet for easier use downstream
# Notes: I used an edited version of the original annotations, where I moved the 'bodily_pain' feature from person_knowledge to social_affective
import os
import numpy as np
import pandas as pd

PROJ_DIR = '/mnt/labdata/got_project'
DATA_DIR = os.path.join(PROJ_DIR, 'ian/data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'features_renamed')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_feature_name_files():
    feat_dir = os.path.join(DATA_DIR, 'feature_names')
    return os.listdir(feat_dir)

def load_feature_names(feat_fn=None):
    feat_dir = os.path.join(DATA_DIR, 'feature_names')
    with open (os.path.join(feat_dir, feat_fn), 'r') as file:
        feat_names = file.readlines()
    no_newlines = [feat.replace('\n', '') for feat in feat_names] # remove newline escape char from extraction
    return no_newlines


def load_features(feat_names=None, sheet_name='true false values edited', transpose=True):
    # sheet_name can be either
    # 'true false values edited'
    # 'social interaction'
    feature_xlsx = os.path.join(DATA_DIR, 'GOT Features.xlsx')
    feat_df = pd.read_excel(feature_xlsx, sheet_name=sheet_name, index_col=0)
    if transpose == True:
        return feat_df.loc[feat_names, :].T
    else:
        return feat_df.loc[feat_names, :]

def recode_perceptual_features():
    test_feats = load_feature_names('perceptual_features.txt')
    test_df = load_features(test_feats)
    new_names = [
        'faces',
        'body_parts',
        'indoor',
        'new_place',
        'round_objects',
        'animate_objects',
        'bgm',
        'written_words'
    ]
    test_df.columns = new_names
    out_fn = os.path.join(OUTPUT_DIR, 'perceptual_features.tsv')
    test_df.to_csv(out_fn, sep='\t')
    return test_df

def recode_soc_aff_features():
    test_feats = load_feature_names('social_affective_features.txt')
    test_df = load_features(test_feats)
    test_df.drop('type of action (fighting, talking, hugging)', axis=1, inplace=True) # drop non-numerical columns for now
    new_names = [
        'social_interaction',
        'talking',
        'talking_self',
        'talking_others',
        'talking_things',
        'inter_person_actions',
        'mentalization',
        'valence',
        'arousal',
        'bodily_pain',
    ]
    test_df.columns = new_names
    out_fn = os.path.join(OUTPUT_DIR, 'social_affective_features.tsv')
    test_df.to_csv(out_fn, sep='\t')
    return test_df


def recode_person_knowledge_features():
    test_feats = load_feature_names('person_knowledge_features.txt')
    test_df = load_features(test_feats)
    new_names = [
        'num_characters',
        'new_character_present',
        'social_hierarchy_present',
    ]

    test_df.columns = new_names
    out_fn = os.path.join(OUTPUT_DIR, 'person_knowledge_features.tsv')
    test_df.to_csv(out_fn, sep='\t')
    return test_df


def recode_character_knowledge():
    test_feats = load_feature_names('character_knowledge_features.txt')
    test_df = load_features(test_feats, sheet_name='social interaction', transpose=False)
    original_colnames = list(test_df.columns)
    mask = [i for i, col in enumerate(original_colnames) if 'average' in str(col)]
    avg_df = test_df.iloc[:, mask].T

    new_index = np.arange(0, len(mask))
    avg_df.set_index(new_index, inplace=True)

    new_names = [
        'character_positivity',
        'positive_behavior',
        'positive_relationship',
        'positivity_past',
        'positivity_future',
        'situation_present',
        'situation_past',
        'situation_future',
    ]
    avg_df.columns = new_names

    out_fn = os.path.join(OUTPUT_DIR, 'character_knowledge_features.tsv')
    avg_df.to_csv(out_fn, sep='\t')

def _testing():
    test_feats = load_feature_names('social_affective_features.txt')
    test_df = load_features(test_feats)
    # print(test_df.columns)
    print(test_df.shape)
    

if __name__ == "__main__":
    # _testing()
    recode_perceptual_features()
    recode_soc_aff_features()
    recode_person_knowledge_features()
    recode_character_knowledge()
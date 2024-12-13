import numpy as np
import pandas as pd
import re
    
    
def get_filename(imaging_id, file_format='ann'):
    """
    Return the filename of the annotation file.
    v1.pet from 13.12.23 (reduced functionality)
    """
    return "report_" + str(imaging_id) + "." + file_format     


def read_reports(x, path):
    """
    Import report texts from .txt files.
    v1 from 13.12.23
    """
    # Define filename
    filename = get_filename(x.imaging_id, file_format='txt')
    
    # Open and read text file
    with open(path + filename, 'r') as f:
        text = f.read()
    
    return text    
        
        
def parse_ann_files(df, path):
    """
    Parse .ann files and write into a dataframe of gold standard concepts.
    v2.pet from 11.01.24
    """
    # Create dataframes to store annotations
    concepts = pd.DataFrame(columns=['imaging_id', 'patient_id', 'scan_no', 
                                     'concept_id', 'concept', 'phrase', 'position'])
    relations = pd.DataFrame(columns=['imaging_id', 'patient_id', 'scan_no', 
                                      'relation_id', 'relation', 'arg1', 'arg2'])

    for _, x in df.iterrows():
        # Define filename
        filename = get_filename(x.imaging_id, file_format='ann')

        # Open and read annotation file
        with open(path + filename, 'r') as f:
            annotation = f.readlines()

        if annotation:    
            # Loop over each line of the annotation file
            for line in annotation:

                # Concept
                if re.match("T", line):

                    # Create an entry containing concept ID, category, position and the raw text
                    substrings = line.strip().split('\t')
                    concept_id = substrings[0]
                    concept = substrings[1].split(maxsplit=1)[0]
                    position = substrings[1].split(maxsplit=1)[1]
                    text = substrings[2]

                    tmp = pd.DataFrame({
                        'imaging_id': x.imaging_id,
                        'patient_id': x.patient_id, 
                        'scan_no': x.scan_no, 
                        'concept_id': concept_id, 
                        'concept': concept, 
                        'phrase': text,
                        'position': position, 
                    }, index=[0])

                    # Add to the table of concepts
                    concepts = pd.concat([concepts, tmp], axis=0, ignore_index=True)

                # Relation
                elif re.match("R", line):

                    # Create an entry containing relation ID, type and IDs of the arguments
                    substrings = line.strip().split()
                    relation_id = substrings[0]
                    relation = substrings[1]
                    arg1 = substrings[2].split(':')[1]
                    arg2 = substrings[3].split(':')[1]

                    tmp = pd.DataFrame({
                        'imaging_id': x.imaging_id,
                        'patient_id': x.patient_id, 
                        'scan_no': x.scan_no, 
                        'relation_id': relation_id, 
                        'relation': relation, 
                        'arg1': arg1, 
                        'arg2': arg2
                    }, index=[0])

                    # Add to the table of relations
                    relations = pd.concat([relations, tmp], axis=0, ignore_index=True)

    # Convert patient ID and report number to int
    concepts[['patient_id', 'scan_no']] = concepts[['patient_id', 'scan_no']].astype(int)
    relations[['patient_id', 'scan_no']] = relations[['patient_id', 'scan_no']].astype(int)
    
    # Start and end character positions
    concepts['start_char'] = concepts.position.apply(lambda x: x.split()[0]).astype(int)
    concepts['end_char'] = concepts.position.apply(lambda x: x.split()[-1]).astype(int)
    
    # Discont concepts have ;-separated positions
    idx = concepts[concepts.position.str.contains(";")].index

    # Flag discont concepts that should be merged into one
    concepts.loc[idx, 'to_merge'] = concepts.iloc[idx].apply(
        lambda x: x.end_char - x.start_char == len(x.phrase), axis=1)

    print("Found %d discontinous concepts that should be merged" % concepts.to_merge.sum())

    # Overwrite position
    concepts.loc[concepts.to_merge==True, 'position'] = concepts[concepts.to_merge==True].apply(
        lambda x: str(x.start_char) + " " + str(x.end_char), axis=1)
    
    # Remove flag
    concepts.drop('to_merge', axis=1, inplace=True)

    print("Extracted %d concepts and %d relations." % (concepts.shape[0], relations.shape[0]))
    
    return concepts, relations


def handle_discont_concepts(concepts):
    """
    Split discontinuos concepts.
    v1 from 13.12.23
    """
    # Discont concepts have ;-separated positions
    idx = concepts[concepts.position.str.contains(";")].index

    # Split discont concepts into a separate dataframe
    discont = concepts.iloc[idx].copy()
    concepts.drop(idx, inplace=True)
    
    # Loop over discont concepts extracting individual spans
    for _,x in discont.iterrows():
        spans = []
        i = 0
        for pos in x.position.split(';'):
            # Extract start and end char positions
            start_char, end_char = map(int, pos.split())
            # Calculate span length
            len_span = end_char - start_char
            # Extract span text
            phrase = x.phrase[i:i+len_span]
            # Add to list of spans
            spans.append((start_char, end_char, phrase))
            i = i + len_span + 1

        # Sort extracted spans by starting position
        spans = sorted(spans, key=lambda x: x[0])

        # Append extracted spans to the dataframe with gold standard concepts 
        for span in spans:
            tmp = x.copy()
            tmp['start_char'] = span[0]
            tmp['end_char'] = span[1]
            tmp['phrase'] = span[2]
            concepts = pd.concat([concepts, tmp.to_frame().T], axis=0, ignore_index=True)

    # Remove position column
    concepts.drop('position', axis=1, inplace=True)
    
    print("After handling discontinous concepts, there are a total of %d concepts." % concepts.shape[0])
    
    return concepts


def adjust_position(x):
    """
    Shift position indices by header length and adjust using the mapping.
    v2 from 15.12.23 (reduced functionality)
    """        
    # Adjust position indices
    start_char = x.pos_mapping.index(x.start_char)
    end_char = x.pos_mapping.index(x.end_char-1) + 1
    
    return start_char, end_char


def get_cue_order(concepts, relations, cues_to_check):
    """
    Check if a cue is preceeding or following.
    v1 from 13.12.23
    """
    def assign_termset(x):
        arg2_ids = relations[(relations.imaging_id==x.imaging_id) & 
                             (relations.arg1==x.concept_id)
                            ].arg2
        arg2_start_char = concepts[(concepts.imaging_id==x.imaging_id) & 
                                    concepts.concept_id.isin(arg2_ids)
                                   ].start_char
        return (x.start_char < arg2_start_char).any(), (x.start_char > arg2_start_char).any()

    # Only check order for positive and negative cue
    cues = concepts[concepts.concept.isin(cues_to_check)]
    
    # Determine if a cue is preceding and/or following
    return pd.DataFrame(cues.apply(assign_termset, axis=1).tolist(), 
                                                        index=cues.index)


def add_composite_concepts(concepts, relations, relations_to_add):
    """
    Combine concepts and relations into composite concepts.
    v2 from 11.01.24
    """
    def get_composite_concept(x, cues):
        # Determine the subject (Arg1) of the relation
        y = concepts[(concepts.imaging_id==x.imaging_id) & (concepts.concept_id==x.arg1)]

        if y.concept.item() in cues.keys():

            # Define the next vacant concept ID
            next_id = concepts[concepts.imaging_id==x.imaging_id].concept_id.apply(lambda x: int(x[1:])).max() + 1
            
            # Determine the object (Arg2) of the relation. 
            # In case of discont concepts, the noun is usually at the end of the phrase: select the last part
            z = concepts[(concepts.imaging_id==x.imaging_id) & 
                         (concepts.concept_id==x.arg2)
                        ].sort_values(by='start_char').iloc[-1]
            
            # Create an entry containing concept ID, composite category, position and the raw text
            return pd.DataFrame({'imaging_id': x.imaging_id,
                                 'patient_id': x.patient_id,
                                 'scan_no': x.scan_no, 
                                 'concept_id': 'T' + str(next_id), 
                                 'concept': cues[y.concept.item()] + '_' + z.concept,
                                 'phrase': z.phrase,
                                 'start_char': z.start_char,
                                 'end_char': z.end_char,
                                }, index=[0])
            
    for k,v in relations_to_add.items():
        # Loop over the dataframe with extracted relations
        for _, x in relations[relations.relation==k].iterrows():
            # Add to the table of concepts
            concepts = pd.concat([concepts, get_composite_concept(x, v)], axis=0, ignore_index=True)
    
     # Drop duplicated composite concepts
    concepts.drop_duplicates(subset=['imaging_id', 'concept', 'start_char'], inplace=True, ignore_index=True)

    print("Totalling %d concepts and composite concepts." % concepts.shape[0])
        
    return concepts


def read_annotations(df, path):
    """
    Parse and post-process .ann files.
    v3.pet from 11.01.24
    """
    # Parse annotation files
    concepts, relations = parse_ann_files(df, path)
    
    # Separate discontinuous concepts
    concepts = handle_discont_concepts(concepts)
    
    # Add information about position changes
    concepts = concepts.merge(df[['imaging_id', 'pos_mapping']], 
                              on='imaging_id')
    
    # Adjust character positions
    concepts[['start_char', 'end_char']] = pd.DataFrame(concepts.apply(adjust_position, axis=1).tolist(), 
                                                        index=concepts.index)
    
    # Drop information about character position shifting
    concepts.drop('pos_mapping', axis=1, inplace=True)
    
    # Preceding and following termsets
    concepts[['preceding', 'following']] = get_cue_order(concepts, relations, ['positive', 'negative'])
    
    # Create composite concepts
    composite_concepts = add_composite_concepts(concepts, relations, 
                                                {'certainty-rel': {'positive': 'affirmed', 
                                                                   'negative': 'negated'}})
    
    # Save the extracted concepts and relations
    concepts.to_csv("../datasets/gold_concepts.csv", index=False)
    relations.to_csv("../datasets/gold_relations.csv", index=False)
    composite_concepts.to_csv("../datasets/gold_composite.csv", index=False)

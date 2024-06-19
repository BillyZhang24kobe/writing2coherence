
# Data format:

The data is in json format and contains a key with the story_Id. 
Each story data is in a dictionary format with the data:
* storyID - integer with the story id 
* Title - string of the story title
* Text - a string of the story text
* HolisticData - a dictionary with the holistic acqired data. The data is: 
    * num_annotators - integer of how many workers annotated in this methods
    * scores: a dict of the scores for each annotator 
        for example: 'scores': {'annot0': 29.0, 'annot1': 41.0}
* IncrementalData - a dictionary with the incremental acquired data. The data is:
    * sentences - a list of the sentences
    * num_annotators - integer of how many worker annotated in this method
    * scores: a dict of the scores for each annotator. the key is the annotatorId and the value is the score. 
        for example: 'scores': {'annot0': 29.0, 'annot1': 41.0}
    * reasons: a dict of the reasons for incoherence of each annotator
        for each annotatorId there is a reason dictionary. The reason dictionary keys are the sentenceId and the value is a list of reason indexed
        for example: {'annot0': {4: [1], 5: [1, 2]}, 'annot1': {4: [1], 5: [1]}}}
        

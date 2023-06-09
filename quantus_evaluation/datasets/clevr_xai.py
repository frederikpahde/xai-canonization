import random
from torch.utils.data import Dataset
from PIL import Image
import re

import numpy as np
import pickle
from tqdm import tqdm
import os
import json
import torch
from torchvision import transforms

classes = {
            'number':['0','1','2','3','4','5','6','7','8','9','10'],
            'material':['rubber','metal'],
            'color':['cyan','blue','yellow','purple','red','green','gray','brown'],
            'shape':['sphere','cube','cylinder'],
            'size':['large','small'],
            'exist':['yes','no']
        }

def collate_samples_from_pixels(batch):
    return collate_samples(batch, False, False)
    
def collate_samples(batch, state_description, only_images):
    """
    Used by DatasetLoader to merge together multiple samples into one mini-batch.
    """
    batch_size = len(batch)

    if only_images:
        images = batch
    else:
        images = [d['image'] for d in batch]
        gts = [torch.Tensor(d['gt_single']) for d in batch]
        answers = [d['answer'] for d in batch]
        questions = [d['question'] for d in batch]
        len_q = [d['len_q'] for d in batch]
        questions_text = [d['question_text'] for d in batch]
        answers_text = [d['answer_text'] for d in batch]
        qid = [d['qid'] for d in batch]
        path_rel_text_precomputed = [d['path_rel_text_precomputed'] for d in batch]
        # questions are not fixed length: they must be padded to the maximum length
        # in this batch, in order to be inserted in a tensor
        # max_len = max(map(len, questions))
        max_len = 43

        padded_questions = torch.LongTensor(batch_size, max_len).zero_()
        for i, q in enumerate(questions):
            padded_questions[i, :len(q)] = q

        
    if state_description:
        max_len = 12
        #even object matrices should be padded (they are variable length)
        padded_objects = torch.FloatTensor(batch_size, max_len, images[0].size()[1]).zero_()
        for i, o in enumerate(images):
            padded_objects[i, :o.size()[0], :] = o
        images = padded_objects
    
    if only_images:
        collated_batch = torch.stack(images)
    else:
        collated_batch = dict(
            image=torch.stack(images),
            gt_single=torch.stack(gts),
            answer=torch.stack(answers),
            question=padded_questions,
            len_q=len_q,
            qid=qid,
            question_text=questions_text,
            answer_text=answers_text,
            path_rel_text_precomputed=path_rel_text_precomputed
        )

    # print("SHAPES", collated_batch['image'].shape, collated_batch['gt_single'].shape)
    return collated_batch


def to_dictionary_indexes(dictionary, sentence):
    """
    Outputs indexes of the dictionary corresponding to the words in the sequence.
    Case insensitive.
    """
    split = tokenize(sentence)
    idxs = torch.LongTensor([dictionary[w] for w in split])
    return idxs

def tokenize(sentence):
    sentence = sentence.lower() # lowercase
    for i in [r'\?',r'\!',r'\-',r'\'',r'\"',r'\$',r'\:',r'\@',r'\(',r'\)',r'\/',r'\,',r'\.',r'\;']: # remove all punctuation
        sentence = re.sub( i, '', sentence)

    # punctuation should be separated from the words
    s = re.sub('([.,;:!?()])', r' \1 ', sentence)
    s = re.sub('\s{2,}', ' ', s)

    # tokenize
    split = s.split()

    # normalize all words to lowercase
    lower = [w.lower() for w in split]
    return lower

def get_transform(img_size=128):
    return transforms.Compose([transforms.Resize((img_size, img_size)), 
                               transforms.ToTensor()])

def correct_answer_type(answer):
    if isinstance(answer, bool):
        return 'yes' if answer else 'no'
    elif isinstance(answer, int):
        return str(answer)
    else:
        return answer

class ClevrXAIDataset(Dataset):
    def __init__(self, dataset_dir, img_size, vocabularies, transform=None, correct_only=False, pred_path='', question_type='simple', shuffle=True):
        assert question_type in ["simple", "complex"], f"question_type {question_type} is invalid."
        
        quest_json_filename = os.path.join(dataset_dir, f"CLEVR-XAI_{question_type}_questions.json")
        self.img_dir = os.path.join(dataset_dir, 'images')
        
        gt_mask_by_question_type = {'simple': 'ground_truth_simple_questions_single_object',
                                    'complex': 'ground_truth_complex_questions_union'}
        
        self.mask_dir = os.path.join(dataset_dir, gt_mask_by_question_type[question_type])
        with open(quest_json_filename, 'r') as json_file:
            self.questions = json.load(json_file)['questions']
            
        ## If there is 'exist' question and answer is 'no', drop the sample
        for q in self.questions:
            q['answer'] = correct_answer_type(q['answer'])
        self.questions = [sample for sample in self.questions if not (('exist' in [pitem['type'] for pitem in sample['program']]) and (sample['answer'] == 'no'))]
        self.questions = [sample for sample in self.questions if os.path.isfile(f"{self.mask_dir}/{sample['question_index']}.npy")]

        if shuffle:
            random.seed(41)
            random.shuffle(self.questions)

        if correct_only:
            correct_labels = np.load(pred_path)
            self.questions = [q for q, correct in zip(self.questions, correct_labels) if correct]

        self.transform=transform
        self.vocabularies=vocabularies
        self.img_size = img_size
        self.dataset_dir = dataset_dir

    def get_class_name_by_index_dict(self):
        return {index: name for name, index in self.vocabularies[1].items()}

    def get_vocabularies(self):
        return self.vocabularies

    def get_denormalizer(self):
        return None

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        current_question = self.questions[idx]
        img_filename = os.path.join(self.img_dir, current_question['image_filename'])
        image = Image.open(img_filename).convert('RGB')
        path_mask = f"{self.mask_dir}/{current_question['question_index']}.npy"
        gt = np.expand_dims(np.load(path_mask).astype(int), axis=0)

        question = to_dictionary_indexes(self.vocabularies[0], current_question['question'])
        answer = to_dictionary_indexes(self.vocabularies[1], current_question['answer'])[0] # - 1

        sample = {'image': image, 
                  'gt_single': gt, 
                  'question': question, 
                  'answer': answer,
                  'qid': current_question['question_index'],
                  'len_q': len(question),
                  'question_text': current_question['question'], 
                  'answer_text': current_question['answer'],
                  'path_rel_text_precomputed': f"{self.dataset_dir}/R_text_precomputed/qid_{current_question['question_index']}.npy"
                }

        if self.transform:
            sample['image'] = self.transform(sample['image'])
        
        return sample


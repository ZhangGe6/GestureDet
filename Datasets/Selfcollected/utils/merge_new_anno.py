import os
import json

hand_anno_path = '../hand_anno.json'
new_hand_anno_path = '../new_hand_anno.json'
object_anno_path = '../object_anno.json'
new_hand_anno_path = '../new_object_anno.json'

# hand
with open(hand_anno_path, 'r') as f:
    anno = json.load(f)
with open(new_hand_anno_path, 'r') as f:
    new_anno = json.load(f)

print('[HAND]: existed {} samples, new {} samples'.format(len(anno['samples']), len(new_anno['samples'])))

anno['samples'] += new_anno['samples']
with open(hand_anno_path, 'w') as f:
    json.dump(anno, f)
print('[HAND]: After merging, there are {} samples now'.format(len(anno['samples'])))


# object
with open(hand_anno_path, 'r') as f:
    anno = json.load(f)
with open(new_hand_anno_path, 'r') as f:
    new_anno = json.load(f)

print('[OBJECT]: existed {} samples, new {} samples'.format(len(anno['samples']), len(new_anno['samples'])))

anno['samples'] += new_anno['samples']
with open(hand_anno_path, 'w') as f:
    json.dump(anno, f)
print('[OBJECT]: After merging, there are {} samples now'.format(len(anno['samples'])))



     

import numpy as np
import numpy.linalg as LA

class GestureClassifier():
    def __init__(self):
        self.version = 0 

    def pred_gesture(self, joints):
        def accu_angle_to_state(accu_angle):
            if accu_angle > 480:
                return 'straight'
            else:
                return 'bending'

        def get_finger_states(joints):
            finger_states = dict()
            thum_angle = calc_accu_angle([joints[0]] + joints[1:5])
            index_finger_angle = calc_accu_angle([joints[0]] + joints[5:9])
            middle_finger_angle = calc_accu_angle([joints[0]] + joints[9:13])
            ring_finger_angle = calc_accu_angle([joints[0]] + joints[13:17])
            pinkie_angle = calc_accu_angle([joints[0]] + joints[17:21])

            finger_states['thumb'] = {'accu_angle' : thum_angle, 'state' : accu_angle_to_state(thum_angle)}
            finger_states['index_finger'] = {'accu_angle' : index_finger_angle, 'state' : accu_angle_to_state(index_finger_angle)}
            finger_states['middle_finger'] = {'accu_angle' : middle_finger_angle, 'state' : accu_angle_to_state(middle_finger_angle)}
            finger_states['ring_finger'] = {'accu_angle' : ring_finger_angle, 'state' : accu_angle_to_state(ring_finger_angle)}
            finger_states['pinkie'] = {'accu_angle' : pinkie_angle, 'state' : accu_angle_to_state(pinkie_angle)}

            return finger_states

        finger_states_dict = get_finger_states(joints)
        finger_accu_angle_list = [finger_states_dict[finger]['accu_angle'] for finger in finger_states_dict.keys()]
        finger_states_list = [finger_states_dict[finger]['state'] for finger in finger_states_dict.keys()]
        
        if finger_states_list[1:] == ['bending', 'straight', 'straight', 'straight'] and dist(joints[4], joints[8]) < 20:
            return 'draw'
        elif finger_states_list[1:] == ['bending', 'bending', 'bending', 'bending']:
            return 'reset'
        else:
            return 'unkonwn gesture'


def calc_angle(vec_a, vec_b):
    vec_a, vec_b = np.array(vec_a), np.array(vec_b)
    inner = np.inner(vec_a, vec_b)
    norms = LA.norm(vec_a) * LA.norm(vec_b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    
    return deg

def calc_accu_angle(pts):
    assert(len(pts) == 5)
    pts = np.array(pts)

    vec0 = pts[1] - pts[0]
    vec1 = pts[2] - pts[1]
    vec2 = pts[3] - pts[2]
    vec3 = pts[4] - pts[2]

    angle1 = calc_angle(-vec0, vec1)
    angle2 = calc_angle(-vec1, vec2)
    angle3 = calc_angle(-vec2, vec3)

    accum_angle = angle1 + angle2 + angle3

    return accum_angle


def dist(joint_a, joint_b):
    # use max rather than L2 norm for simper calculation 
    return max(abs(joint_a[0] - joint_b[0]), abs(joint_a[1] - joint_b[1]))

import numpy as np 
import matplotlib.pyplot as plt

sm = {
    'blank': 0,
    'wall': 1,
    'hole': 2,
    'oil': 3,
    'start': 4,
    'goal': 5
}

oil_coords = [
    (2,8), (2,16), (4,2), (5,6), (10,18), 
    (15,10), (16,10), (17,14), (17,17), (18,7)
]

hole_coords = [
    (1, 11), (1,12), (2,1), (2,2), (2,3),
    (5,1), (5,9), (5,17), (6,17), (7,2), (7,10),
    (7,11), (7,17), (8,17), (12,11), (12,12), 
    (14,1), (14,2), (15,17), (15,18), (16,7)
]

start_coord = (15, 4)
end_coord = (3, 13)

def build_map():
    m = np.zeros((20,20))

    # Do wall with fancy indexing since it's mostly lines
    # Borders
    m[0] = sm['wall']
    m[19] = sm['wall']
    m[:, 0] = sm['wall']
    m[:, 19] = sm['wall']

    # Inner walls
    m[2:5, 5] = sm['wall']
    m[4, 3:17] = sm['wall']
    m[4:8, 3] = sm['wall']
    m[6:13, 6] = sm['wall']
    m[6:11, 9] = sm['wall']
    m[6:12, 15] = sm['wall']
    m[7, 12:16] = sm['wall']
    m[10, 1:5] = sm['wall']
    m[10:15, 10] = sm['wall']
    m[11:16, 13] = sm['wall']
    m[11:14, 17] = sm['wall']
    m[11, 16] = sm['wall']
    m[12, 3:8] = sm['wall']
    m[12:16, 7] = sm['wall']
    m[15, 13:17] = sm['wall']
    m[17, 1:3] = sm['wall']
    m[17, 7:13] = sm['wall']

    # Individual coords
    for oc in oil_coords:
        m[oc] = sm['oil']
    for hc in hole_coords:
        m[hc] = sm['hole']

    m[start_coord] = sm['start']
    m[end_coord] = sm['goal']

    return m

def visualize(m):
    plt.imshow(m)
    plt.show()

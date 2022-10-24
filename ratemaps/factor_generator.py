
import numpy as np


def get_related_head_angle(vec1, vec2):
    """

    :param vec1: record animal allo head direction vector from top view
    :param vec2: perform animal allo head direction or vector from neck to reward (base)
    :return: signed angle from vec1 to vec2 in degree
    """

    vec = np.array([-vec2[1], vec2[0]])
    b_coord = np.dot(vec1, vec2)
    p_coord = np.dot(vec1, vec)
    da_ang = np.arctan2(p_coord, b_coord)

    return da_ang


def get_related_dist(marker1, marker2):
    """

    :param marker1: record animal neck marker
    :param marker2: perform animal neck marker or reward
    :return: distance of marker1 and marker2 from top view
    """
    dist2 = np.sqrt(np.sum((marker1[:2] - marker2[:2]) ** 2))
    return dist2


def relative_neck_elevation(point_data):
    """
    neck elevation of one animal related to the Ass marker
    :return:
    """
    neck_pnts = point_data[:, 4, :]
    butt_pnts = point_data[:, 6, :]

    nf = len(neck_pnts)

    elev = np.zeros(nf)
    elev[:] = np.nan

    for t in range(nf):
        if np.isnan(neck_pnts[t, 0]) or np.isnan(butt_pnts[t, 0]):
            continue

        elev[t] = neck_pnts[t, 2] - butt_pnts[t, 2]

    return elev


def get_related_head(rotm1, rotm2, use_solution_with_least_tilt=False):
    """

    :param rotm1: record animal allo head rotation matrix
    :param rotm2: perform animal allo head rotation matrix
    :return: rrotm: related head rotm, how to rotate record
    """
    n_frames = len(rotm1)
    if len(rotm2) != n_frames:
        raise Exception('check data, 2 rotm must be same length')

    rrotm = np.zeros((n_frames, 3, 3))
    rrotm[:] = np.nan
    rangs = np.zeros((n_frames, 3))
    rangs[:] = np.nan

    for t in range(n_frames):
        if np.isnan(rotm1[t, 0, 0]) or np.isnan(rotm2[t, 0, 0]):
            continue

        rhx = np.dot(rotm2[t], rotm1[t, 0, :])
        rhy = np.dot(rotm2[t], rotm1[t, 1, :])
        rhz = np.dot(rotm2[t], rotm1[t, 2, :])

        rrotm[t] = np.array([rhx, rhy, rhz])
        rangs[t] = rot2angles(rrotm[t], useexpmap=False, use_xzy_order=False, usesolutionwithleasttilt=use_solution_with_least_tilt)

    return rrotm, rangs


def angle_between_2vectors(vector1, vector2):
    unit_vector_1 = vector1 / np.linalg.norm(vector1)
    unit_vector_2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def get_related_body_direction(pdata, rdata):
    """
    angles between body direction vectors from top view (2d), and
    angles between body direction vectors in 3d
    body direction from ass marker to neck marker
    :return:
    """
    nf = len(pdata['body_direction'])
    ppnts = pdata['sorted_point_data'].copy()
    rpnts = rdata['sorted_point_data'].copy()

    angs2d = np.zeros(nf)
    angs2d[:] = np.nan

    angs3d = np.zeros(nf)
    angs3d[:] = np.nan

    for t in np.arange(nf):
        check_pnts = np.array([ppnts[t, 4, 0], ppnts[t, 6, 0], rpnts[t, 4, 0], rpnts[t, 6, 0]])
        if np.any(np.isnan(check_pnts)):  # 4 is neck and 6 is tail
            continue

        p_body_vec = ppnts[t, 4, :] - ppnts[t, 6, :]
        p_body_vec_2d = p_body_vec.copy()
        p_body_vec_2d[2] = 0

        r_body_vec = rpnts[t, 4, :] - rpnts[t, 6, :]
        r_body_vec_2d = r_body_vec.copy()
        r_body_vec_2d[2] = 0

        angs3d[t] = angle_between_2vectors(p_body_vec, r_body_vec)
        angs2d[t] = angle_between_2vectors(p_body_vec_2d, r_body_vec_2d)

    return angs2d, angs3d
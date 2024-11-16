from face_3dmm.bfm import BaselFaceModel
from face_3dmm.fvm import FaceVerseModel
from face_3dmm.flame import FLAMEModel

MODELS = ['fvm', 'bfm', 'bfm+fw_exp', 'flame']

def load_face_model(**kwargs):
    import pickle
    import numpy as np
    from scipy.io import loadmat
    from pytorch3d.io.obj_io import load_obj

    if kwargs['model'] == 'fvm':
        pdict = np.load('assets/faceverse_v3_6_s.npy', allow_pickle=True).item()
        model = FaceVerseModel(**pdict, **kwargs)
    elif kwargs['model'] == 'bfm':
        pdict = loadmat('assets/BFM_model_front.mat')
        model = BaselFaceModel(**pdict, img_size=kwargs['img_size'])
    elif kwargs['model'] == 'bfm+fw_exp':
        pdict = loadmat('assets/BFM_model_front.mat')
        pdict['exBase'] = np.load('assets/mean_delta_blendshape.npy')
        model = BaselFaceModel(**pdict, **kwargs)
    elif kwargs['model'] == 'flame':
        with open('assets/generic_model.pkl', 'rb') as f:
            pdict = pickle.load(f, encoding='latin1')
        with open('assets/FLAME_masks.pkl', 'rb') as f:
            pdict.update(pickle.load(f, encoding='latin1'))
        pdict.update(**np.load('assets/landmark_embedding.npy', allow_pickle=True, encoding='latin1').item())
        pdict.update(**np.load('assets/FLAME_texture.npz'))
        faces, aux = load_obj('assets/head_template.obj')[1:]
        pdict.update({'uv_coords': aux.verts_uvs, 'uv_tris': faces.textures_idx})
        model = FLAMEModel(**pdict, **kwargs)
    else:
        raise NotImplementedError(f"Unsupported 3DMM model: {kwargs['model']}!")
    return model

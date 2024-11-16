import torch
import torch.nn.functional as F

from face_3dmm import MODELS
from face_3dmm import load_face_model

from face_reconstruction.regression.network import PENet, PENetWrapper
from face_reconstruction.regression.thirdlibs.cpem.network import ResNet50_3DMM
from face_reconstruction.regression.thirdlibs.deca.network import ResnetEncoder

from utils import apply_imagenet_normalization

SUPPORTED_THIRD_LIBS = ['cpem', 'deca']
SUPPORTED_MODELS = MODELS + SUPPORTED_THIRD_LIBS

def smooth_coeffs(smooth):

    prev_coeffs = []

    def smooth_coeffs_inner(coeffs):
        prev_coeffs.append(coeffs)
        if len(prev_coeffs) > smooth:
            prev_coeffs.pop(0)
        return sum(prev_coeffs) / len(prev_coeffs)

    return smooth_coeffs_inner

def prepare_pipeline(**kwargs):
    assert(kwargs['model'] in SUPPORTED_MODELS), f"Unsupported model: {kwargs['model']}!"

    if kwargs['model'] == 'cpem':
        model = load_face_model(model='bfm+fw_exp', img_size=kwargs['img_size'])
        network = ResNet50_3DMM()

        try:
            checkpoint = torch.load(kwargs['checkpoint'])
            network.load_state_dict(checkpoint['network'])
            print(f"Successfully loaded checkpoint {kwargs['checkpoint']}.")
        except Exception as e:
            if 'checkpoint' in kwargs.keys():
                print(f"Exception occured when loading {kwargs['checkpoint']}: {str(e)}")
            checkpoint = torch.load('assets/resnet50-id-exp-300000.ckpt')
            network.load_state_dict(checkpoint)
            print(f"Loaded the default CPEM checkpoint.")

        preprocess = lambda x: F.interpolate(x, size=(224, 224))
        postprocess = smooth_coeffs(kwargs.get('smooth', 1))
    elif kwargs['model'] == 'deca':
        model = load_face_model(model='flame', img_size=kwargs['img_size'], id_dim=100, exp_dim=50, tex_dim=50)
        network = ResnetEncoder(outsize=236).to('cuda:0')

        try:
            checkpoint = torch.load(kwargs['checkpoint'])
            network.load_state_dict(checkpoint['network'])
            print(f"Successfully loaded checkpoint {kwargs['checkpoint']}.")
        except Exception as e:
            if 'checkpoint' in kwargs.keys():
                print(f"Exception occured when loading {kwargs['checkpoint']}: {str(e)}")
            checkpoint = torch.load('assets/deca_model.tar')
            network.load_state_dict(checkpoint['E_flame'])
            print(f"Loaded the default DECA checkpoint.")

        def freeze_neck_eye_pose(coeffs):
            id_coeff = coeffs[:, :100]
            tex_coeff = coeffs[:, 100: 150]
            exp_coeff = coeffs[:, 150: 200]
            body_pose = coeffs[:, 200: 203]
            jaw_pose = coeffs[:, 203: 206]
            camera = coeffs[:, 206: 209]
            gamma = coeffs[:, 209:]

            neck_pose = model.get_default_neck_pose(coeffs.shape[0], coeffs.device)
            eye_pose = model.get_default_eye_pose(coeffs.shape[0], coeffs.device)
            return model.merge_coeffs(id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma)

        preprocess = lambda x: F.interpolate(x, size=(224, 224))
        postprocess = [freeze_neck_eye_pose, smooth_coeffs(kwargs.get('smooth', 1))]
    else:
        model = load_face_model(**kwargs)
        network = PENet(out_dim=model.all_dim)

        try:
            checkpoint = torch.load(kwargs['checkpoint'])
            network.load_state_dict(checkpoint['network'])
            print(f"Successfully loaded checkpoint from {kwargs['checkpoint']}.")
        except Exception as e:
            if 'checkpoint' in kwargs.keys():
                print(f"Exception occured when loading {kwargs['checkpoint']}: {str(e)}")

        def freeze_neck_pose(coeffs):
            coeffs_dict = model.split_coeffs(coeffs, return_dict=True)
            coeffs_dict['neck_pose'] = model.get_default_neck_pose(coeffs.shape[0], coeffs.device)
            return model.merge_coeffs(**coeffs_dict)

        preprocess = [lambda x: F.interpolate(x, size=kwargs['img_size'])]
        preprocess.append(lambda x: apply_imagenet_normalization(x))
        postprocess = [freeze_neck_pose] if kwargs['model'] == 'flame' else []
        postprocess.append(smooth_coeffs(kwargs.get('smooth', 1)))

    return model, PENetWrapper(network, preprocess, postprocess)

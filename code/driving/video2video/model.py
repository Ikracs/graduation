import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained.headpose_estimation import HopeNet
from driving.video2video.utils import ImagePyramide, Vgg19
from driving.video2video.utils import bins2degree
from driving.video2video.utils import make_coordinate_grid_like_3d

from utils import requires_grad


class GeneratorFullModel(nn.Module):
    def __init__(self, generator, discriminator, train_params):
        super(GeneratorFullModel, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.pyramid = ImagePyramide(train_params.scales, generator.img_channel)

        if sum(train_params.loss_weights.perceptual) != 0:
            self.vgg = Vgg19()
            self.vgg.eval()
            requires_grad(self.vgg, False)

        if train_params.loss_weights.headpose != 0:
            self.hopenet = HopeNet()
            self.hopenet.eval()
            requires_grad(self.hopenet, False)

        self.train_params = train_params

    def forward(self, x):
        requires_grad(self.generator, True)
        requires_grad(self.discriminator, False)

        generated = self.generator(x['src_img'], x['dri_img'])

        pyramide_real = self.pyramid(x['dri_img'])
        pyramide_generated = self.pyramid(generated['prediction'])

        loss_values = {}

        if sum(self.train_params.loss_weights.perceptual) != 0:
            value_total = 0
            for scale in self.train_params.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)][:, :3])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)][:, :3])

                for i, weight in enumerate(self.train_params.loss_weights.perceptual):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += weight * value
            loss_values['perceptual'] = value_total

        if self.train_params.loss_weights.generator != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)

            value_total = 0
            for scale in self.discriminator.scales:
                key = 'prediction_map_%s' % scale
                if self.train_params.gan_loss == 'hinge':
                    value = -discriminator_maps_generated[key].mean()
                elif self.train_params.gan_loss == 'ls':
                    value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                else:
                    raise ValueError('Unexpected gan_loss {}'.format(self.train_params.gan_loss))

                value_total += self.train_params.loss_weights.generator * value
            loss_values['generator'] = value_total

            if sum(self.train_params.loss_weights.feature_matching) != 0:
                value_total = 0
                for scale in self.discriminator.scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.train_params.loss_weights.feature_matching[i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.train_params.loss_weights.feature_matching[i] * value
                    loss_values['feature_matching'] = value_total

        if self.train_params.loss_weights.reconstruction != 0:
            generated['reconstruction'] = self.generator.decoder(generated['feature'])
            value = torch.abs(x['src_img'] - generated['reconstruction']).mean()
            loss_values['reconstruction'] = self.train_params.loss_weights.reconstruction * value

        if self.train_params.loss_weights.id_consistency != 0:
            cid_dri_id = self.generator.id_extractor(x['cid_img'])
            cid_generated = self.generator(x['src_img'], x['cid_img'])
            cid_generated_id = self.generator.id_extractor(cid_generated['prediction'])

            generated['cid_prediction'] = cid_generated['prediction']

            pos_sim = 5 * (F.cosine_similarity(generated['source_id'], cid_generated_id) - 0.2)
            neg_sim = 5 * (F.cosine_similarity(cid_dri_id, cid_generated_id) - 0.2)

            value = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
            loss_values['id_consistency'] = self.train_params.loss_weights.id_consistency * value

        if self.train_params.loss_weights.motion_consistency != 0:
            generated_exp = self.generator.pm_estimator(generated['prediction'])['motion']

            pos_sim = 5 * (F.cosine_similarity(generated['driving_motion'], generated_exp) - 0.2)
            neg_sim = 5 * (F.cosine_similarity(generated['source_motion'], generated_exp) - 0.2)

            value = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
            loss_values['motion_consistency'] = self.train_params.loss_weights.motion_consistency * value

        if self.train_params.loss_weights.headpose != 0:
            yaw, pitch, roll = [bins2degree(i) for i in self.hopenet(x['dri_img'])]
            value = torch.abs(yaw - generated['driving_pose']['yaw']).mean()
            value += torch.abs(pitch - generated['driving_pose']['pitch']).mean()
            value += torch.abs(roll - generated['driving_pose']['roll']).mean()
            loss_values['headpose'] = self.train_params.loss_weights.headpose * value

        if self.train_params.loss_weights.mask != 0:
            value = torch.abs(x['src_msk'] - generated['source_mask']).mean()
            value += torch.abs(x['dri_msk'] - generated['driving_mask']).mean()
            loss_values['mask'] = self.train_params.loss_weights.mask * value

        if self.train_params.loss_weights.regularization != 0:
            identity_grid = make_coordinate_grid_like_3d(generated['feature'])
            value = torch.norm(generated['deformation'] - identity_grid, p=1, dim=-1).mean()
            loss_values['regularization'] = self.train_params.loss_weights.regularization * value

        return loss_values, generated


class DiscriminatorFullModel(nn.Module):
    def __init__(self, generator, discriminator, train_params):
        super(DiscriminatorFullModel, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.pyramid = ImagePyramide(train_params.scales, generator.img_channel)

        self.train_params = train_params

    def forward(self, x, generated):
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        pyramide_generated = self.pyramid(generated['prediction'].detach())
        pyramide_real = self.pyramid(x['dri_img'])

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}

        value_total = 0
        for scale in self.discriminator.scales:
            key = 'prediction_map_%s' % scale
            if self.train_params.gan_loss == 'hinge':
                value = -torch.min(discriminator_maps_real[key] - 1, torch.zeros_like(discriminator_maps_real[key])).mean()
                value -= torch.min(-discriminator_maps_generated[key] - 1, torch.zeros_like(discriminator_maps_generated[key])).mean()
            elif self.train_params.gan_loss == 'ls':
                value = ((1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2).mean()
            else:
                raise NotImplementedError('Unexpected gan_loss {}'.format(self.train_params.gan_loss))

            value_total += self.train_params.loss_weights.discriminator * value
        loss_values['discriminator'] = value_total

        return loss_values

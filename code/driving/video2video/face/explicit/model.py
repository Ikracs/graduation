import torch
import torch.nn as nn

from pretrained.headpose_estimation import HopeNet
from driving.video2video.utils import ImagePyramide, Vgg19
from driving.video2video.utils import make_coordinate_grid_like_3d

from utils import requires_grad


class GeneratorFullModel(nn.Module):
    def __init__(self, penet, generator, discriminator, train_params):
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

        if train_params.loss_weights.coeff_match != 0:
            self.penet = penet
            self.penet.eval()
            requires_grad(self.penet, False)

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

        if self.train_params.loss_weights.coeff_match != 0:
            value = torch.abs(generated['source_coeffs'] - self.penet(x['src_img'])).mean()
            value += torch.abs(generated['driving_coeffs'] - self.penet(x['dri_img'])).mean()
            loss_values['coeff_match'] = self.train_params.loss_weights.coeff_match * value

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

# Author: Jacek Komorowski
# Warsaw University of Technology
import torch
import models.minkloc as minkloc


def model_factory(params):
    in_channels = 32 if params.use_unet else 1

    if 'MinkFPN' in params.model_params.model:
        model = minkloc.MinkLoc(params.model_params.model, in_channels=in_channels,
                                feature_size=params.model_params.feature_size,
                                output_dim=params.model_params.output_dim, planes=params.model_params.planes,
                                layers=params.model_params.layers, num_top_down=params.model_params.num_top_down,
                                conv0_kernel_size=params.model_params.conv0_kernel_size, use_unet = params.use_unet, fix_frontbone = params.fix_frontbone)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(params.model_params.model))

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if params.fix_frontbone and params.use_unet:
        model.frontend.load_state_dict(torch.load(params.frontbone_weight, map_location=device)["state_dict"])


    return model

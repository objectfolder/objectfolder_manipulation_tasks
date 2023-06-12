import numpy as np
import torch
import torch.nn as nn

from hydra.utils import to_absolute_path

from model_predictive_control.models.model import VideoPredictionModel
from model_predictive_control.mpc.utils import slice_dict, cat_dicts


class SVGPrime(nn.Module):
    """
    SVG' model as a nn.Module, assuming that all modules are already created.
    See "from_config" to construct a new SVGPrime model with fresh weights.
    """

    def __init__(self, encoder, decoder, frame_predictor, posterior, prior):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.frame_predictor = frame_predictor
        self.posterior = posterior
        self.prior = prior

    def init_hidden(self, batch_size):
        # Set batch sizes for hidden states
        self.frame_predictor.batch_size = batch_size
        self.posterior.batch_size = batch_size
        self.prior.batch_size = batch_size

        # Initialize hidden states
        self.frame_predictor.hidden = self.frame_predictor.init_hidden()
        self.posterior.hidden = self.posterior.init_hidden()
        self.prior.hidden = self.prior.init_hidden()

    def forward(self, x, actions):
        """
        :param x: input context images in shape [T, B, C, H, W]
        :param actions: input actions in shape [T, B, a_dim]
        :return: predicted video in shape [T, B, C, H, W]
        """
        from svg_prime.utils import tile_actions_into_image
        batch_size = x.shape[1]
        self.init_hidden(batch_size=batch_size)
        gen_seq = list()
        x_in = x[0]
        num_context = len(x)
        for i in range(1, len(actions) + 1):
            h = self.encoder(x_in)
            if i < num_context:
                h, skip = h
            else:
                h, _ = h
            tiled_action = tile_actions_into_image(actions[i - 1], h.shape[-2:])
            if i < num_context:
                h_target = self.encoder(x[i])
                h_target = h_target[0]
                z_t, _, _ = self.posterior(h_target)
                self.prior(h)
                self.frame_predictor(torch.cat([h, tiled_action, z_t], dim=1))
                x_in = x[i]
                gen_seq.append(x_in)
            else:
                z_t, _, logvar = self.prior(h)
                h = self.frame_predictor(torch.cat([h, tiled_action, z_t], dim=1))
                x_in = self.decoder([h, skip])
                gen_seq.append(x_in)
        return torch.stack(gen_seq)

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError("TODO")

class MultiSVG(nn.Module):
    def __init__(self, model_v, model_t):
        super().__init__()
        self.model_v = model_v
        self.model_t = model_t

    def forward(self, x, actions):
        """
        :param x: input context images in shape [T, B, C, H, W]
        :param actions: input actions in shape [T, B, a_dim]
        :return: predicted video in shape [T, B, C, H, W]
        """
        width = int(x.shape[-1] / 2)
        v_input = x[:, :, :, :, :width]
        t_input = x[:, :, :, :, width:]
        v_pred = self.model_v(v_input, actions)
        t_pred = self.model_t(t_input, actions)
        
        return torch.cat((v_pred, t_pred), dim=-1)
        
class SVGPrimeInterface(VideoPredictionModel):
    def __init__(
        self,
        checkpoint_dir,
        n_past,
        planning_modalities,
        max_batch_size=400,
        epoch=None,
    ):
        if len(checkpoint_dir) == 2:
            checkpoint_dir_v = checkpoint_dir[0]
            checkpoint_dir_t = checkpoint_dir[1]
            self.checkpoint_file_v = self.get_checkpoint_file(checkpoint_dir_v, epoch)
            saved_model_v = torch.load(self.checkpoint_file_v)
            self.checkpoint_file_t = self.get_checkpoint_file(checkpoint_dir_t, epoch)
            saved_model_t = torch.load(self.checkpoint_file_t)
            
            self.model = MultiSVG(
                SVGPrime(
                encoder=saved_model_v["encoder"],
                decoder=saved_model_v["decoder"],
                frame_predictor=saved_model_v["frame_predictor"],
                posterior=saved_model_v["posterior"],
                prior=saved_model_v["prior"],
                ), 
                SVGPrime(
                encoder=saved_model_t["encoder"],
                decoder=saved_model_t["decoder"],
                frame_predictor=saved_model_t["frame_predictor"],
                posterior=saved_model_t["posterior"],
                prior=saved_model_t["prior"],
            ))
        else:
            checkpoint_dir_any = checkpoint_dir[0]
            self.checkpoint_file = self.get_checkpoint_file(checkpoint_dir_any, epoch)
            saved_model = torch.load(self.checkpoint_file)
        
            self.model = SVGPrime(
                encoder=saved_model["encoder"],
                decoder=saved_model["decoder"],
                frame_predictor=saved_model["frame_predictor"],
                posterior=saved_model["posterior"],
                prior=saved_model["prior"],
            )
        self.model.eval()
        self.model.cuda()
        self.planning_modalities = planning_modalities
        self.base_prediction_modality = "rgb"
        self.num_context = n_past
        self.max_batch_size = max_batch_size

    def format_model_epoch_filename(self, epoch):
        return f"model_{epoch}.pth"

    def prepare_batch(self, xs):
        keys = ["video", "actions"]
        batch = {
            k: torch.from_numpy(x).cuda().float() for k, x in xs.items() if k in keys
        }
        batch["video"] = torch.permute(batch["video"], (1, 0, 4, 2, 3))
        batch["actions"] = torch.permute(batch["actions"], (1, 0, 2))
        return batch["video"], batch["actions"]

    def __call__(self, batch):
        all_preds = list()
        with torch.no_grad():
            video, actions = self.prepare_batch(batch)
            for compute_batch_idx in range(
                0, batch["video"].shape[0], self.max_batch_size
            ):
                predictions = self.model(
                    video[
                        :, compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                    actions[
                        :, compute_batch_idx : compute_batch_idx + self.max_batch_size
                    ],
                )
                predictions = predictions.permute(1, 0, 3, 4, 2)
                predictions = predictions.cpu().numpy()
                all_preds.append(predictions)
        predictions = np.concatenate(all_preds, axis=0)
        return dict(rgb=predictions)

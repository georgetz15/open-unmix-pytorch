import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import model
import utils
import warnings
import tqdm
from contextlib import redirect_stderr
import io
import os
import sys
import math
mdensenet_path = os.path.expanduser('~/repos/music-source-separation/source/models/')
sys.path.append(mdensenet_path)

import MMDenseNet as mdn


def load_model(target, model_name='umxhq', device='cpu', model_type='mdensenet'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        state = torch.load(
            target_model_path,
            map_location=device
        )

        if model_type == 'mdensenet':
            unmix = mdn.MDenseNet(growth_rate=12, block_config=(4, 4, 4), compression=.5,
                                  num_init_features=32, bn_size=2, drop_rate=0.1,
                                  efficient=True, final_growth_rate=4, num_final_layers=2).to(device)
            unmix.load_state_dict(state)
            unmix.mdensenet.stft.center = True
        elif model_type == 'open-unmix':
            max_bin = utils.bandwidth_to_max_bin(
                state['sample_rate'],
                results['args']['nfft'],
                results['args']['bandwidth']
            )
            unmix = model.OpenUnmix(
                n_fft=results['args']['nfft'],
                n_hop=results['args']['nhop'],
                nb_channels=results['args']['nb_channels'],
                hidden_size=results['args']['hidden_size'],
                max_bin=max_bin
            )
            unmix.load_state_dict(state)
            unmix.stft.center = True

        unmix.eval()
        unmix.to(device)
        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
    audio,
    targets,
    model_name='umxhq',
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu',
    model_type='mdensenet'
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    model_type: string
        Which model to use: `open-unmix` or `mdensenet`

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device,
            model_type=model_type
        )
        if model_type == 'mdensenet':
            with torch.no_grad():
                x = unmix_target.mdensenet.transform(audio_torch)
                num_frames = math.ceil((3.0 * 44100 - 4096) / 1024)
                X = torch.split(x, num_frames, dim=3)
                Vj = []
                for i in range(len(X)):
                    Vj.append(unmix_target(X[i]))
                Vj = torch.cat(Vj, dim=3).cpu().detach().numpy()
            Vj = np.transpose(Vj, [3, 0, 1, 2])
            n_fft = unmix_target.mdensenet.stft.n_fft
            n_hop = unmix_target.mdensenet.stft.n_hop
        elif model_type == 'open-unmix':
            Vj = unmix_target(audio_torch).cpu().detach().numpy()
            n_fft = unmix_target.stft.n_fft
            n_hop = unmix_target.stft.n_hop


        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    if model_type == 'mdensenet':
        X = unmix_target.mdensenet.stft(audio_torch).detach().cpu().numpy()
    elif model_type == 'open-unmix':
        X = unmix_target.stft(audio_torch).detach().cpu().numpy()
    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X[0].transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)

    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=n_fft,
            n_hopsize=n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=int,
        default=1,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'input',
        type=str,
        nargs='+',
        help='List of paths to wav/flac files.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals', 'drums', 'bass', 'other'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Audio chunk start in seconds'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=-1.0,
        help='Audio chunk duration in seconds, negative values load the full track'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    for input_file in args.input:
        # handling an input audio path
        info = sf.info(input_file)
        start = int(args.start * info.samplerate)
        # check if dur is none
        if args.duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(args.duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(
            input_file,
            always_2d=True,
            start=start,
            stop=stop
        )

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != args.samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, args.samplerate, axis=0)

        if audio.shape[1] == 1:
            # if we have mono, let's duplicate it
            # as the input of OpenUnmix is always stereo
            audio = np.repeat(audio, 2, axis=1)

        estimates = separate(
            audio,
            targets=args.targets,
            model_name=args.model,
            niter=args.niter,
            alpha=args.alpha,
            softmask=args.softmask,
            residual_model=args.residual_model,
            device=device
        )
        if not args.outdir:
            model_path = Path(args.model)
            if not model_path.exists():
                outdir = Path(Path(input_file).stem + '_' + args.model)
            else:
                outdir = Path(Path(input_file).stem + '_' + model_path.stem)
        else:
            outdir = Path(args.outdir)
        outdir.mkdir(exist_ok=True, parents=True)

        # write out estimates
        for target, estimate in estimates.items():
            sf.write(
                outdir / Path(target).with_suffix('.wav'),
                estimate,
                args.samplerate
            )

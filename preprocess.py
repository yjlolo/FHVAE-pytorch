import glob
import os
import librosa
import numpy as np
import math
import argparse


def _process_utterance(out_dir, index, wav_path):
    if not os.path.exists(os.path.join(out_dir)):
        os.makedirs(os.path.join(out_dir))
    if not os.path.exists(os.path.join(out_dir, str(index))):
        os.makedirs(os.path.join(out_dir, str(index)))
    # Load the audio to a numpy array:
    wav, sr = librosa.load(wav_path, sr=16000)

    wav = wav / np.abs(wav).max() * 0.999
    out = wav
    constant_values = 0.0
    out_dtype = np.float32
    n_fft = 640
    hop_length = 160

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    spectrogram = librosa.core.stft(wav, n_fft=n_fft, hop_length=hop_length, center=False).T

    log_spectrum = librosa.core.amplitude_to_db(np.abs(spectrogram))
    log_spectrum = (log_spectrum - log_spectrum.min()) / (log_spectrum.max() - log_spectrum.min())
    pad = (out.shape[0] // hop_length + 1) * hop_length - out.shape[0]
    pad_l = pad // 2
    pad_r = pad // 2 + pad % 2

    # zero pad for quantized signal
    out = np.pad(out, (pad_l, pad_r), mode="constant", constant_values=constant_values)

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:out.shape[0] // hop_length * hop_length]

    timesteps = len(out)
    num_spec = log_spectrum.shape[0] // 20

    # Write the spectrograms to disk:
    audio_filename = '{}/audio.npy'.format(index)
    np.save(os.path.join(out_dir, audio_filename), out.astype(out_dtype))
    for i in range(num_spec):
        spectrum_filename = '{}/log-mag-spectrum-{}.npy'.format(index, i)
        np.save(os.path.join(out_dir, spectrum_filename), log_spectrum.astype(np.float32)[20 * i : 20 * (i + 1), :])


def output_to_wav(output_spectrum):
    spec_amplitude = librosa.core.db_to_amplitude(output_spectrum)
    wav_recon = reconstruct_signal_griffin_lim(spec_amplitude)
    return wav_recon


def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size=640, hopsamp=160, iterations=1000):
    """Reconstruct an audio signal from a magnitude spectrogram.
    Given a magnitude spectrogram as input, reconstruct
    the audio signal and return it using the Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
    Args:
        magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
            and the columns correspond to frequency bins.
        fft_size (int): The FFT size, which should be a power of 2.
        hopsamp (int): The hope size in samples.
        iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
            is sufficient.
    Returns:
        The reconstructed time domain signal as a 1-dim Numpy array.
    """
    magnitude_spectrogram = magnitude_spectrogram.T
    time_slices = magnitude_spectrogram.shape[1]
    len_samples = int((time_slices - 1) * hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = librosa.core.stft(x_reconstruct, n_fft=fft_size, hop_length=hopsamp, center=False)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram * np.exp(1.0j * reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = librosa.core.istft(proposal_spectrogram, win_length=fft_size, hop_length=hopsamp, center=False)
        diff = math.sqrt(sum((x_reconstruct - prev_x) ** 2) / x_reconstruct.size)
        #print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', '-i', type=str, default='./timit/TIMIT', help='In Directory')
    parser.add_argument('--out_dir', '-o', type=str, default='./', help='Out Directory')
    parser.add_argument('--option', type=str, default='all', help='Out Directory')
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.option == 'all':
        dirlist = glob.glob(os.path.join(args.in_dir, '*/*/*/*.WAV'))
    elif args.option == 'train':
        dirlist = glob.glob(os.path.join(args.in_dir, 'TRAIN', '*/*/*.WAV'))
    elif args.option == 'test':
        dirlist = glob.glob(os.path.join(args.in_dir, 'TEST', '*/*/*.WAV'))
    else:
        pass

    for i, wav_path in enumerate(dirlist):
        _process_utterance(args.out_dir, i, wav_path)
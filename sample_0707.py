import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def main():
    y, sr = librosa.load(librosa.util.example_audio_file())
    amplitude = np.abs(librosa.stft(y))
    matrix_fc, matrix_ct = librosa.decompose.decompose(amplitude, n_components=16, sort=True)
    reconstructed_amplitude = matrix_fc.dot(matrix_ct)

    fig = plt.figure(figsize=(6.4*2, 4.8*2))

    ax1 = fig.add_subplot(3, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(amplitude, ref=np.max), y_axis='log', x_axis='time')
    ax1.set_title('Input spectrogram')
    plt.colorbar(format='%+2.0f dB')

    ax2 = fig.add_subplot(3, 2, 3)
    librosa.display.specshow(librosa.amplitude_to_db(matrix_fc, ref=np.max), y_axis='log')
    ax2.set_xlabel('Components')
    plt.colorbar(format='%+2.0f dB')
    ax2.set_title('Matrix_fc')

    ax3 = fig.add_subplot(3, 2, 4)
    librosa.display.specshow(matrix_ct, x_axis='time')
    ax3.set_ylabel('Components')
    ax3.set_title('Matrix_ct')
    plt.colorbar()

    ax4 = fig.add_subplot(3, 1, 3)
    librosa.display.specshow(librosa.amplitude_to_db(reconstructed_amplitude, ref=np.max), y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    ax4.set_title('Reconstructed spectrogram')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
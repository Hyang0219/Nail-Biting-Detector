import numpy as np
import wave
import struct
import os

def generate_beep():
    # Parameters for the beep sound
    duration = 0.5  # seconds
    frequency = 440.0  # Hz (A4 note)
    sample_rate = 44100  # Hz
    amplitude = 0.5  # relative amplitude (0.0 to 1.0)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate sine wave
    signal = amplitude * np.sin(2.0 * np.pi * frequency * t)
    
    # Convert to 16-bit integer samples
    samples = [int(sample * 32767) for sample in signal]
    
    # Get the absolute path to the resources directory
    resources_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources')
    os.makedirs(resources_dir, exist_ok=True)
    
    # Create and configure the wave file
    wave_path = os.path.join(resources_dir, 'alert.wav')
    with wave.open(wave_path, 'w') as wave_file:
        # Set parameters
        nchannels = 1  # mono
        sampwidth = 2  # 2 bytes per sample (16-bit)
        framerate = sample_rate
        nframes = len(samples)
        comptype = 'NONE'
        compname = 'not compressed'
        
        wave_file.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
        
        # Write the samples
        for sample in samples:
            wave_file.writeframes(struct.pack('h', sample))

if __name__ == '__main__':
    generate_beep() 
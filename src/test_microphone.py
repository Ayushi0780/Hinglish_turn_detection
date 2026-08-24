import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5

print("=" * 60)
print("LAPTOP MICROPHONE TEST")
print("=" * 60)

print()
print("Your laptop microphone will record for 5 seconds.")
print("Speak normally during the recording.")
print()

input("Press ENTER to start recording...")

print()
print("🎤 RECORDING NOW — SPEAK!")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print()
print("✓ Recording finished.")

audio = audio[:, 0]

rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print()
print("=" * 60)
print("MICROPHONE RESULTS")
print("=" * 60)

print(f"Duration       : {DURATION} seconds")
print(f"Average volume : {rms:.6f}")
print(f"Peak volume    : {peak:.6f}")

sf.write(
    "test_recording.wav",
    audio,
    SAMPLE_RATE
)

print()
print("✓ Saved recording:")
print("  test_recording.wav")

if rms < 0.001:
    print()
    print("⚠️ VERY LOW AUDIO")
    print("The microphone may not be capturing your voice.")

else:
    print()
    print("✓ Microphone is capturing audio.")

print("=" * 60)
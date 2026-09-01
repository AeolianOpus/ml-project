import sounddevice as sd

print("All input devices:\n")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        hostapi = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i:2d}] {d['name']}  ({d['max_input_channels']} ch)  [{hostapi}]")

print("\nAll output devices:\n")
for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        hostapi = sd.query_hostapis(d["hostapi"])["name"]
        print(f"  [{i:2d}] {d['name']}  ({d['max_output_channels']} ch)  [{hostapi}]")
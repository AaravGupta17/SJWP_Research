"""
╔══════════════════════════════════════════════════════╗
║  MENDELEY MANUAL VALIDATION GUIDE                    ║
╠══════════════════════════════════════════════════════╣
║  Your Mendeley data is in:                           ║
║    ../datasets/Hydrophone/                           ║
║    ../datasets/Accelerometer/                        ║
║    ../datasets/Dynamic Pressure Sensor/              ║
║                                                      ║
║  To validate visually:                               ║
║  1. Load a Mendeley hydrophone signal (real data)    ║
║  2. Load a synthetic signal from your training cache ║
║  3. Compare frequency spectra (FFT) — both should    ║
║     show energy concentrated below 1kHz              ║
║  4. Compare TDOA: cross-correlate the two channels   ║
║     of a real leak signal — should show a clear      ║
║     peak offset, matching synthetic TDOA range       ║
║                                                      ║
║  Run: python mendeley_check.py                       ║
╚══════════════════════════════════════════════════════╝
"""
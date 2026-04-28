HUMAN TRANSCRIPTS FOLDER
========================

Purpose
-------
Put one CSV file per call in this folder.
The file name MUST match the audio file name exactly.

  Audio file:  data/food_malay.wav
  CSV file:    human_transcripts/food_malay.csv

These CSVs are ONLY used by compare_labels.py to check whether
the system's Agent/Customer labels match what you say they should be.
They are NOT used during classification — the system classifies
automatically from audio.

How to fill in the CSV
----------------------
1. Open the audio file and listen
2. For each speaker turn, write down:
   - segment_id   : a sequential number (1, 2, 3...)
   - role         : Agent  or  Customer  (what YOU say this speaker is)
   - text         : approximately what they said
                    (does not need to be perfect — fuzzy matching handles differences)
   - start        : when this segment starts in seconds (optional but helps)
   - end          : when this segment ends in seconds (optional)

Required columns: segment_id, role, text
Optional columns: start, end

Tips
----
- You do NOT need to transcribe every single segment — just enough to cover
  representative turns from both Agent and Customer
- For Manglish/mixed language, write the text as you hear it
- The text does NOT need to match Whisper exactly — similarity of 35%+ is enough
- Check outputs/{call_id}_diarized.json to see what Whisper transcribed,
  then write text similar to that

Example
-------
See example_format.csv in this folder for the correct layout.

After filling in your CSVs, run:
  python compare_labels.py

To compare only one call:
  python compare_labels.py --call_id food_malay

To see every wrong segment printed:
  python compare_labels.py --show_diff

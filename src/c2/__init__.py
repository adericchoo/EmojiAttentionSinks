"""C2 (Isaac): tokenisation audit, probe generator, metric, experiment runner.

Builds on the C1 pipeline in ``eas`` (model loading, attention capture, span
location). Nothing here modifies ``eas``; it imports from it.

Units are kept separate throughout:
  code points  -> Unicode scalar values (U+1F44D)
  graphemes    -> one emoji sequence from emoji-test.txt (may be many code points)
  UTF-8 bytes  -> the encoded bytes of the sequence
  model tokens -> what the tokenizer produces; "fragmentation" always means this.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "outputs", "c2")
DATA_DIR = os.path.join(ROOT, "data")

PILOT_LABEL = "DEVELOPMENT PILOT – not confirmatory results"

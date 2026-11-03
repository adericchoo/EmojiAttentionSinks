"""The 8 carrier families and the fixed neutral text around them.

Every probe is   PREFIX + header + "\\n" + [SLOT] + tail + " " + CONTINUATION
                 |----- fixed, ends in a newline -----|      |--- fixed ---|

Why the slot starts a line: a newline is always its own token in both
tokenizers, so the tokens BEFORE the slot are identical whatever the filler is.
With a space before the slot, byte-level BPE glues that space onto the first
filler token for some fillers (most emoji) but not others (rare CJK in Pythia,
digits in Qwen), which would shift the probe's start index between conditions.

Why the tail starts with a space: a space always starts a new pre-token, so the
tokens AFTER the slot are identical too (checked for every probe anyway).

Each header gives the slot a neutral role: a value being reported (a label,
an ID, a cell). Any filler (emoji, word, rare characters, hex) reads as equally
natural or equally odd there. Fillers do not need to make sense.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Carrier:
    name: str
    header: str   # line before the slot
    tail: str     # rest of the slot's line; must start with a space


CARRIERS = [
    Carrier("label", "The storage box on the top shelf has this label:",
            " written on a strip of tape across the lid."),
    Carrier("list_item", "The checklist pinned to the door starts with this entry:",
            " followed by a short note saying that it was checked on Monday."),
    Carrier("quoted_marker", "In the margin of the draft, the editor left this marker, quoted exactly:",
            " (copied from page four of the printed draft)."),
    Carrier("id_field", "Archive record 2214, field ID:",
            " | status: closed | owner: records office | last checked in March."),
    Carrier("annotation", "A reviewer added one annotation next to the second paragraph:",
            " and then moved on to the next section without further comment."),
    Carrier("tag", "The shared folder was saved with the following tag:",
            " so that the team could find it again during the next review."),
    Carrier("table_cell", "Row four of the inventory table, column B, contains:",
            " while the other cells in that row were left empty by the night shift."),
    Carrier("subject_line", "The subject line of the archived email reads:",
            " and the body of the message is only two lines long."),
]
CARRIER_NAMES = [c.name for c in CARRIERS]

# Neutral text BEFORE the carrier. Trimmed word by word (per model, per carrier)
# so that the probe starts at exactly the target token index.
PREFIX = (
    "This page is part of a set of ordinary office notes. The notes were typed up at the end "
    "of each week and stored in a shared folder on the main computer. Most of them describe "
    "small tasks, such as ordering paper, fixing a printer, booking a room, or moving desks "
    "between rooms on the second floor. None of the notes is urgent, and most of them were "
    "written quickly by whoever happened to be at the front desk that afternoon. The entry "
    "below was copied without changes from an older file, which had been kept in a drawer "
    "near the front desk for several months before it was finally scanned and added to the "
    "folder with the other weekly notes from that year. Nothing else in the file was changed."
)

# Neutral text AFTER the carrier tail: supplies the 16-token buffer and the
# 64-token query block Q. Identical for every probe.
CONTINUATION = (
    "The rest of the document describes a routine meeting held on Thursday afternoon. The team "
    "reviewed the schedule for the coming weeks, agreed on who would book the rooms, and "
    "discussed the budget for printing and stationery. Several people asked about the new "
    "parking arrangements, and the manager promised to send a short update by email before the "
    "end of the week. After a brief coffee break, the group went through the list of open tasks "
    "one by one, marking each as done, in progress, or postponed until the next month. The "
    "meeting ended a few minutes early, and everyone returned to their desks to finish the day."
)

"""Every constant the tactile compiler and BRF writer lay out with, each with its source (all fetched 2026-10-24).

Sources:
  TG    BANA, "Guidelines and Standards for Tactile Graphics, 2010", web manual,
        https://www.brailleauthority.org/tg/web-manual/tgmanual.html (section numbers quoted per constant).
  TG-F  TG Appendix F "Line Styles", image https://www.brailleauthority.org/tg/web-manual/images/appFline-styles.jpg
  TG-E  TG Appendix E "Texture Palette for Microcapsule Paper", image .../images/appEtexture-palette.jpg
        ((c) 2010 RNIB, used by BANA/CBA; we only name textures from it, the drawings are ours).
  SIZE  BANA, "Size and Spacing of Braille Characters",
        https://www.brailleauthority.org/size-and-spacing-braille-characters, quoting Library of Congress (NLS)
        Specification 800 "Braille Books and Pamphlets", February 2008, section 3.2 (nominal values for paper).
  FMT   BANA, "Braille Formats: Principles of Print-to-Braille Transcription, 2016",
        https://www.brailleauthority.org/formats/2016manual-web/section01.html and section11.html
  UEB   ICEB, "The Rules of Unified English Braille", 2024,
        https://iceb.org/Rules%20of%20Unified%20English%20Braille%202024.pdf

Gridlines: none are drawn. TG 6.6.1.1 "Some grid lines may be omitted if not necessary to the understanding of the
graph"; TG 6.6.4.2 / 6.6.6.3 suggest ADDING them when the reader must track values, so that is a reviewer's call.
All lengths are inches; line weights are points (1/72 in) as TG-F prints them.
"""

PT = 1 / 72  # inches per point

# --- Braille cell (SIZE: NLS Spec 800, section 3.2) ---
DOT_DIAMETER = 0.057  # 3.2.2 "The nominal base diameter of braille dots shall be 0.057 inches [1.44 mm]"
DOT_SPACING = 0.092   # 3.2.3.1 center to center of adjacent dots (horizontally or vertically) in the same cell
CELL_SPACING = 0.245  # 3.2.3.2 center to center of corresponding dots in adjacent cells
LINE_SPACING = 0.400  # 3.2.4 line spacing, center to center of nearest corresponding dots in adjacent lines
# Swell paper: TG Appendix D.2 says with a standard braille font "the braille dots become too large and difficult to
# read" once developed, and recommends Duxbury's Swell Braille font, without a printed-dot size. UNVERIFIED knob:
# calibrate on the fuser in use.
SWELL_DOT_SCALE = 1.0

# --- Page (TG 3.4.2, 3.4.4.2; FMT 1.7.1) ---
# FMT 1.7.1 "Materials are usually embossed on 11 1/2 x 11 braille paper with 25 lines, with a maximum of 40 cells per
# line"; TG 3.4.4.2 graphics use "the same overall paper size as the text of the book, usually 11 by 11-1/2 inches".
# One orientation: the graphic is capped at 40 cells x 25 lines (9.8 x 10 in) either way.
PAGE_W, PAGE_H = 11.5, 11.0
CELLS_PER_LINE = 40  # TG 3.4.2.1 "Maximum overall width of tactile graphic: 40 cells"
LINES_PER_PAGE = 25  # TG 3.4.2.2 "Maximum overall length of tactile graphic: 25 lines" (title and key lines included)

# --- Lines (TG-F weights) ---
AXIS_W = 2.5 * PT  # TG-F "Axis line (2.5pt)"
TICK_W = 1.5 * PT  # TG-F "Tick mark (1.5pt)"
TICK_HALF = 0.25   # TG 6.6.1.2 tick marks "must be 1/2 inch ... long and shown 1/4 inch ... on either side of a line"
# Plotted lines are "the strongest" lines (TG 6.6.2.2; every weight here exceeds AXIS_W). At most 3 per graphic: a
# product cap inside TG 3.2.2 "no more than ... five different line styles". Weights from TG-F: "Plotted line 2
# (4.0pt)", "Dashed (3.0pt)"; TG-F prints no weight for a dotted line: UNVERIFIED 4pt dots.
LINE_STYLES = (
    {"name": "solid line", "w": 4.0 * PT, "dash": None},
    {"name": "dashed line", "w": 3.0 * PT, "dash": (0.3, 0.15)},  # TG 3.4.3.3 dash 1/4 to 3/8 in, gaps about half
    {"name": "dotted line", "w": 4.0 * PT, "dash": (0.0, 0.15)},  # round caps make the dots; 0.15 in pitch UNVERIFIED
)
MAX_LINES_PER_PAGE = len(LINE_STYLES)
MIN_LINE = 0.5  # TG 3.4.3.2 "Primary lines must be minimum 1/2 inch (1.25 centimeters) in length"

# --- Point symbols (TG 3.4.3.11, 6.6.2.2, 6.6.5.7) ---
POINT_SYMBOLS = ("circle", "square", "triangle")
POINT_SIZE = 0.25    # TG 3.4.3.11 "For discrimination between two or more different-shaped symbols, the minimum
                     # diameter must be at least 1/4 inch"
POINT_MIN = 0.125    # TG 6.6.2.2 "The point should not be smaller than 1/8 inch (3 millimeters) in diameter"
POINT_CLEAR = 0.125  # TG 6.6.2.2 "use a 1/8 inch (3 millimeters) blank space all around a point"

# --- Bars (TG 6.6.6.7, TG-E) ---
BAR_MIN, BAR_MAX = 0.375, 1.0  # TG 6.6.6.7 bars "a minimum of 3/8 inch (1 centimeter) wide ... a maximum of 1 inch"
BAR_OUTLINE = 3.0 * PT  # TG-F "Plotted line 4 (3.0pt)" weight; TG gives no bar outline weight (UNVERIFIED choice)
BAR_GAP = 0.125         # between bar sets TG 6.6.6.7 gives no figure ("separated enough ... close enough to readily
                        # compare"); we use the TG 3.4.3.12 minimum of 1/8 in between components
# Area textures for bar series, at most 4 per graphic (product cap inside TG 3.2.2 "five different area textures").
# TG-E group A "can be used freely, together or with textures from other groups"; group C: "Only use 1 texture".
BAR_TEXTURES = ("none", "diagonal", "dots", "grid")  # none = outline only; diagonal = TG-E C5; dots = A8; grid = A3
TEXTURE_PITCH = 0.125  # hatch line / dot spacing inside textures (grid 1.5x); TG-E prints no dimensions: UNVERIFIED
TEXTURE_DOT = 0.04     # dot diameter of the dots texture: UNVERIFIED, read off the TG-E A8 sample by eye
MAX_BAR_SERIES_PER_PAGE = len(BAR_TEXTURES)

# --- Labels (TG 3.4.3.12, 5.10, 6.6) ---
CLEAR = 0.125          # TG 3.4.3.12 labels "a minimum of 1/8 inch (3 millimeters) ... from any other component";
                       # TG 3.4.3.11 the same 1/8 in around point symbols
LABEL_GAP_MAX = 0.25   # TG 3.4.3.12 "... to a maximum 1/4 inch (6 millimeters)"; TG 5.10.5 same range to its component
MAX_TICK_LABELS = 6    # product cap; TG 6.6.1.3 "some numbers can be omitted from the vertical and horizontal axis
                       # lines ... A transcriber's note explaining this modification is not necessary"
TICK_MARK_MIN_SPACING = 0.375  # closer print ticks keep only their labelled marks: TG 3.4.3.9 grid lines "no closer
                               # than 3/8 inch", applied to ticks by analogy (UNVERIFIED as a tick rule)
AXIS_TITLE_MAX_CELLS = 20  # TG 6.6.4.5 / 6.6.6.6 "If the label is more than 20 cells, divide the label between lines"
GUIDE_DOT = '"'            # TG 6.6.6.6 guide dots are dot 5 (Braille ASCII '"')
MIN_GUIDE_DOTS = 3         # TG 6.6.6.6 "Leave one blank cell before inserting a minimum of three unspaced guide dots"
HBAR_LABEL_MAX_CELLS = 14  # product choice; longer category labels on horizontal bars are keyed
                           # (TG 6.6.6.6 "If values are lengthy, they may be keyed")
SERIES_LABEL_RESERVE = 2.5  # inches kept right of a multi-line plot for direct labels (product choice;
                            # TG 2.17 "use a braille label (word) instead of a keyed symbol when it will fit")

# --- Keys (TG 5.4, 5.7, 5.8, 5.9; UEB 3.27) ---
KEY_AVOID = {"aa", "ac", "cc", "ca"}  # TG 5.8.1.2 "use of aa, ac, cc, and ca ... should be avoided"
# TG 5.8.1.2 "One of the letters in each key entry must include dot 3 and/or dot 6": checked in tactile.Keyer.
KEY_SAMPLE_LEN = 1.0      # TG 5.9.2 line symbols in a key "must be 1 inch (2.5 centimeters) in length"
KEY_AREA = (1.0, 0.5)     # TG 3.4.3.14 / 5.9.1.1-2 area texture sample 1 in wide, 1/2 in high (two braille lines)
KEY_TEXT_CELL = 6         # TG 5.8.4.3 "the explanation starts in cell 6 with runovers in cell 8"
KEY_TEXT_RUNOVER = 8
KEY_LETTER_TEXT_CELL = 4  # TG 5.8.4.5 "Explanations for 2-cell symbols begin in cell 4 ... runovers starting in cell 3"
KEY_LETTER_RUNOVER = 3
KEY_HEAD_CELL = 7         # TG 5.8.4.1 "The word 'Key:' preceded by a transcriber's note symbol in cell 7"
TN_RUNOVER = 5            # TG 5.4.1 transcriber's note "starting in cell 7 ... with runovers in cell 5"
TN_OPEN, TN_CLOSE = "@.<", "@.>"  # UEB 3.27 transcriber's note indicators, dots 4-46-126 and 4-46-345

# --- BRF text (FMT) ---
PARA_INDENT, PARA_RUNOVER = 3, 1    # FMT 1.9.2 "Use 3-1 margins for indented paragraphs"
TABLE_INDENT, TABLE_RUNOVER = 1, 3  # FMT 11.17.1 d, f linear table format "using 1-3 margins"
BLANK_ENTRY = '"""'                 # FMT 11.17.1 k "three unspaced dot 5s for a blank entry"
SEMICOLON = "2"                     # UEB semicolon, dots 23 (liblouis en-ueb-g2: 'cat;' -> 'cat2'; checked in tests)

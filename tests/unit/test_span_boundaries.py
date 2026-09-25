"""Span validation against the REAL stored text shape.

The pilot found this: 26 of 28 span rejections were `starts_mid_sentence` on
perfectly legitimate quotes, demoting 55% of records against a 15% threshold.

Cause: `description_text` joins blocks with "\\n", but the normaliser used for
matching collapses ALL whitespace, destroying the block boundary. The check
then walks back to the previous BLOCK's last character — and Norwegian ads are
bullet lists whose items rarely end in punctuation.

Every earlier test used single-line text with full stops between sentences, so
the production shape was never exercised.
"""
import pytest

from finn_smart_search.understanding import census_validate as v

pytestmark = pytest.mark.unit

# The real shape: html_clean emits blocks, silver joins them with newlines.
# Every span here carries a language token: a span with none is rejected by a
# different rule (no_language_token), which is correct — evidence must be ABOUT
# language — and would mask what these tests are for.
BULLETS = ("Du må beherske norsk godt\n"
           "Har erfaring med søm, enten fra arbeid eller utdanning\n"
           "Gode norskkunnskaper\n"
           "Engelsk er også nyttig i denne stillingen")


def test_a_whole_block_quote_is_valid_evidence():
    """A bullet with no terminal punctuation, preceded by another such bullet.
    This is the commonest evidence shape in the corpus."""
    ok, why = v._span_ok("Gode norskkunnskaper", BULLETS)
    assert ok, f"rejected a legitimate block quote: {why}"


def test_the_first_block_is_valid_evidence():
    ok, why = v._span_ok("Du må beherske norsk godt", BULLETS)
    assert ok, why


def test_the_last_block_is_valid_evidence():
    ok, why = v._span_ok("Engelsk er også nyttig i denne stillingen", BULLETS)
    assert ok, why


def test_a_fragment_of_a_block_is_still_rejected():
    """The 578-ad trap must still be caught: a block boundary is a boundary,
    but mid-block is not. The fragment carries a language token, so it is the
    BOUNDARY rule that must reject it."""
    ok, why = v._span_ok("beherske norsk godt", BULLETS)
    assert not ok and why == "starts_mid_sentence"


def test_a_cross_block_span_is_rejected_as_not_verbatim():
    """DELIBERATE, and it is not a boundary test despite appearances.

    norm() collapses the newline inside the span while norm_keep_blocks() keeps
    it in the text, so NO cross-block span can ever be a substring — the check
    exits at the first rule and never reaches the boundary logic. Asserting the
    reason code is the only thing this test does that the verdict does not.

    Both joinings must report the same code, and that code must not be confused
    with fabrication when the next pilot's rejections are triaged:
    `not_verbatim` here means "quoted across a block break", not "invented".
    A faithful newline-preserving quote of two adjacent blocks lands here too.
    """
    for span in ("Gode norskkunnskaper Engelsk er også nyttig i denne stillingen",
                 "Gode norskkunnskaper\nEngelsk er også nyttig i denne stillingen"):
        assert v._span_ok(span, BULLETS) == (False, "not_verbatim")


def test_sentence_boundaries_still_work_within_a_block():
    text = "Vi søker medarbeider. Gode norskkunnskaper. Oppstart snarest."
    assert v._span_ok("Gode norskkunnskaper.", text)[0]
    assert not v._span_ok("norskkunnskaper. Oppstart", text)[0]


def test_mixed_shape_blocks_and_sentences():
    """Real ads have both: prose blocks containing several sentences, and
    bullet blocks containing none."""
    text = ("Om stillingen\n"
            "Vi søker en medarbeider. Norsk er arbeidsspråket. Oppstart etter avtale.\n"
            "Krav\n"
            "Gode norskkunnskaper")
    assert v._span_ok("Norsk er arbeidsspråket.", text)[0], "sentence inside a prose block"
    assert v._span_ok("Gode norskkunnskaper", text)[0], "whole bullet block"



# ── gaps found by external review of this file ───────────────────────────────
#
# The review's finding: the END-boundary rule was completely unprotected.
# Replacing the whole `ends_mid_sentence` block with `pass` left all 7 tests
# green, and `ends_mid_sentence` was asserted in no test in the repo. The start
# rule was well guarded (two independent mutations to it die); the end rule had
# nothing. That asymmetry matters because making a block edge a legitimate START
# boundary — which is the fix this file exists to pin — leaves the END rule as
# the only defence against quoting the head of a bullet and dropping its
# negation.

NEGATED = ("Vi tilbyr opplæring\n"
           "Gode norskkunnskaper er ikke et krav hos oss\n"
           "Oppstart snarest")


def test_a_bullet_head_with_the_negation_dropped_is_rejected():
    """THE anti-neutering guard, and the most dangerous span in the corpus.

    "Gode norskkunnskaper" starts at a real block boundary, is verbatim, clears
    15 chars and carries a language token — it satisfies every other rule. It
    also asserts the exact OPPOSITE of the sentence it was cut from. Only the
    end-boundary rule rejects it, so without this test that rule can be deleted
    silently and the validator's whole purpose goes with it.
    """
    assert v._span_ok("Gode norskkunnskaper", NEGATED) == (False, "ends_mid_sentence")


def test_trailing_space_before_a_block_break_is_still_a_boundary():
    """_H_SPACE folds the trailing run to ONE space, so k lands on " " and not
    on the newline. Deleting the end-side skip loop rejects every span whose
    block had trailing whitespace — and html_clean cannot be relied on to have
    stripped it in every path."""
    text = "Vi søker en medarbeider   \nGode norskkunnskaper   \nOppstart"
    assert v._span_ok("Gode norskkunnskaper", text) == (True, "")


def test_crlf_block_separator_is_a_boundary():
    """_H_SPACE is `[^\\S\\n]+`, not `[ \\t]+`, and the difference is load-bearing:
    under `[ \\t]+` a \\r survives into the text, is skipped by neither skip loop
    and is not in BOUNDARY, so every span before a CRLF break is rejected."""
    text = "Vi søker en medarbeider\r\nGode norskkunnskaper\r\nOppstart"
    assert v._span_ok("Gode norskkunnskaper", text) == (True, "")


def test_a_blank_line_between_blocks_is_a_boundary():
    text = "Vi søker en medarbeider\n\nGode norskkunnskaper\n\nOppstart"
    assert v._span_ok("Gode norskkunnskaper", text) == (True, "")


def test_a_span_without_a_language_token_is_rejected_for_that_reason():
    """Pins the claim this file's own header makes about its fixtures. Weakening
    or deleting the LANG_TOKEN rule was otherwise free."""
    text = "Om stillingen\nVi tilbyr fleksibel arbeidstid og godt miljø\nKrav"
    assert v._span_ok("Vi tilbyr fleksibel arbeidstid og godt miljø", text) == (
        False, "no_language_token")


def test_a_too_short_span_is_rejected_for_that_reason():
    """`fragment_too_short` was asserted nowhere in the repo, so the 15-char
    floor could be weakened to 5 undetected."""
    assert v._span_ok("norsk", "Krav\nnorsk\nOppstart") == (False, "fragment_too_short")


def test_norm_keep_blocks_folds_everything_except_newlines():
    """The function had no direct test at all, so `.strip()` was free to delete
    and the NFKC/_TRANSLATE folding it re-implements was unchecked. Pinning it
    against the shared normaliser is what stops the two drifting — the exact
    class of bug that produced the 55% demotion."""
    from finn_smart_search.understanding.text_norm import normalise

    assert v.norm_keep_blocks(None) == ""
    assert v.norm_keep_blocks("") == ""
    assert v.norm_keep_blocks("  Gode norskkunnskaper  ") == "Gode norskkunnskaper"
    # newlines survive; horizontal whitespace folds to one space
    assert v.norm_keep_blocks("a\u00a0 b \t c\n\n  d ") == "a b c\n\n d"
    # identical folding to the shared normaliser on text with no newline
    s = "Gode\u00a0norsk\u00adkunnskaper\u2014ja \u201cB1\u201d"
    assert v.norm_keep_blocks(s) == normalise(s)


# ── mutation survivors: two rules the above tests left free ──────────────────

def test_a_span_that_also_occurs_mid_sentence_is_still_accepted():
    """`find()` inspects only the FIRST occurrence, so a span appearing once
    inside a prose sentence and once as its own block was rejected on the
    strength of the occurrence nobody quoted.

    Measured cost before the fix: 6 of 12,572 language-bearing corpus blocks
    (0.05%), across 5 ads. Small — recorded here rather than inflated. It is
    fixed because the rule is wrong, not because the number is large: a span is
    defensible if it faithfully quotes SOME sentence, and `rfind()` is no more
    correct than `find()`. The mutation `find -> rfind` survived the whole file
    before this test existed.
    """
    span = "Gode norskkunnskaper er et krav"
    text = ("Vi forventer Gode norskkunnskaper er et krav hos alle nyansatte\n"
            f"{span}\n"
            "Oppstart snarest")
    assert v._span_ok(span, text) == (True, "")


def test_a_span_occurring_only_mid_sentence_is_still_rejected():
    """The other half — the fix must not become "accept if it appears at all"."""
    span = "Gode norskkunnskaper er et krav"
    text = ("Vi forventer Gode norskkunnskaper er et krav hos alle nyansatte\n"
            "Oppstart snarest")
    assert v._span_ok(span, text) == (False, "starts_mid_sentence")


def test_an_inline_colon_is_a_boundary():
    """Norwegian ads write "Krav: Gode norskkunnskaper" as ONE block when the
    heading is not separately marked up, so ":" carries the boundary. Removing
    ":" from BOUNDARY survived every other test in this file."""
    text = "Om oss\nKrav: Gode norskkunnskaper muntlig og skriftlig\nOppstart"
    assert v._span_ok("Gode norskkunnskaper muntlig og skriftlig", text) == (True, "")


def test_a_semicolon_is_a_boundary():
    text = "Om oss\nVi tilbyr mye; Gode norskkunnskaper er et krav\nOppstart"
    assert v._span_ok("Gode norskkunnskaper er et krav", text) == (True, "")


def test_the_reported_reason_is_the_first_occurrences():
    """A deliberate reporting choice, pinned because reason codes are what made
    the 55%-demotion diagnosis possible: 26 of 28 rejections reading
    `starts_mid_sentence` is what located the bug. When every occurrence fails,
    report the FIRST one's reason — the earliest occurrence is the likeliest to
    be what the model meant to quote. `first_reason = reason` (last wins) is
    otherwise a free mutation.
    """
    span = "Gode norskkunnskaper er et krav"
    # first occurrence fails on the START rule, second on the END rule
    text = (f"Vi forventer {span} hos alle nyansatte\n"
            f"Krav: {span} og god IT-forståelse")
    assert v._span_ok(span, text) == (False, "starts_mid_sentence")


# ── bullet GLYPHS: the sixth hand-written pattern to be the thing at fault ────
#
# The glyph-bullet probe measured "12/22 model accuracy" on ads stating a
# disjunction as a bullet. All 9 misses were DEMOTED, 8 of them
# `starts_mid_sentence`. The model had found and quoted every requirement
# correctly; the validator threw the evidence away and the demotion reset the
# level to `unstated`, which is why every miss looked like the model not seeing
# the line.
#
# Cause: BOUNDARY enumerated • U+2022 but not · U+00B7 (1,930 corpus blocks) or
# ● U+25CF (141). The model quotes the bullet's TEXT without the glyph, so the
# character before the span is the glyph, which was not a boundary.
#
# Why the earlier corpus-wide check said 0 false rejects: it passed whole
# blocks INCLUDING the glyph, so the character before the span was the newline.
# It never reproduced what the model actually does. Same shape of blind spot as
# the span tests that used single-line prose.
#
# 2,998 corpus blocks are led by a glyph absent from BOUNDARY, and the tail is
# unenumerable: middle dot, black circle, 📍 ✅ ✨ 👉 ⭐ 🤝 🔹, U+200B zero-width
# space, U+2060 word joiner, and U+F0B7 -- the Wingdings bullet Microsoft Word
# emits into pasted job ads.
#
# So the rule is not a longer list. Walking back from a span, LIST FURNITURE --
# anything not a letter or digit -- is skipped; a sentence terminator or a block
# edge accepts; a letter or digit rejects. That is robust to glyphs nobody has
# thought of, which is the only property worth having here.

def bullet(glyph: str) -> str:
    return (f"Om stillingen\n{glyph} Gode norskkunnskaper er et krav\n{glyph} Oppstart snarest")


def test_middle_dot_bullet_is_a_boundary():
    """U+00B7, 1,930 corpus blocks — the single commonest glyph absent from the
    old BOUNDARY set, and the one in 6 of the 9 demoted probe ads."""
    assert v._span_ok("Gode norskkunnskaper er et krav", bullet("·")) == (True, "")


def test_black_circle_bullet_is_a_boundary():
    assert v._span_ok("Gode norskkunnskaper er et krav", bullet("●")) == (True, "")


def test_word_joiner_between_glyph_and_text_is_skipped():
    """From ad 7cbe546e, verbatim: '•⁠ ⁠Norsk (Skandinavisk) eller engelsk'.
    U+2060 is a format character, not whitespace, so _H_SPACE does not fold it
    and the horizontal-space skip loop walked straight into it."""
    text = "Krav\n•⁠ ⁠Gode norskkunnskaper er et krav\nOppstart"
    assert v._span_ok("Gode norskkunnskaper er et krav", text) == (True, "")


def test_zero_width_space_before_text_is_skipped():
    """U+200B leads 106 corpus blocks."""
    text = "Krav\n​Gode norskkunnskaper er et krav\nOppstart"
    assert v._span_ok("Gode norskkunnskaper er et krav", text) == (True, "")


def test_wingdings_bullet_from_word_is_a_boundary():
    """U+F0B7, private use area, 29 corpus blocks. Nobody would think to add
    this to a hand-written list -- which is the argument for not keeping one."""
    assert v._span_ok("Gode norskkunnskaper er et krav", bullet("")) == (True, "")


@pytest.mark.parametrize("glyph", ["\U0001F4CD", "✅", "✨", "\U0001F449",
                                   "⭐", "\U0001F91D", "\U0001F539", "✔"])
def test_emoji_bullets_are_boundaries(glyph):
    """📍 ✅ ✨ 👉 ⭐ 🤝 🔹 ✔ all lead corpus blocks. Parametrised so the count of
    covered glyphs is visible rather than buried in one assertion."""
    assert v._span_ok("Gode norskkunnskaper er et krav", bullet(glyph)) == (True, "")


def test_furniture_skipping_does_not_accept_a_real_mid_sentence_span():
    """THE guard. Skipping non-alphanumerics must not turn the start rule into
    always-true: a span preceded by a WORD is still mid-sentence.

    This is the case the whole boundary rule exists for -- 863 corpus ads say
    "norsk og engelsk", so "engelsk" lifted out of one is a real substring
    supporting the opposite of the sentence."""
    text = "Krav\n· Du må beherske norsk og engelsk godt\nOppstart"
    assert v._span_ok("engelsk godt og presist nok", text) == (False, "not_verbatim")
    # and a genuine substring of the sentence, long enough to pass the floor:
    ok, why = v._span_ok("beherske norsk og engelsk godt", text)
    assert (ok, why) == (False, "starts_mid_sentence"), f"got {ok} {why}"


def test_a_digit_before_the_span_still_rejects():
    """Numbered lists: "1. Gode norskkunnskaper" — the "." accepts, but a bare
    digit run must not be treated as furniture in a way that lets
    "3 norsk eller engelsk" through as a boundary."""
    text = "Krav\nSe punkt 3 norsk eller engelsk kreves her\nOppstart"
    assert v._span_ok("norsk eller engelsk kreves her", text) == (False, "starts_mid_sentence")


# ── BOUNDARY conflated two roles (external review, measured) ──────────────────
#
# One set served both ends of the span, but the two roles are not the same:
#
#   what may PRECEDE a span   `.!?:;` and a block edge — a colon legitimately
#                             ends a lead-in ("Krav: Gode norskkunnskaper")
#   what may TERMINATE a span `.!?` and a block edge ONLY — a colon, semicolon
#                             or hyphen does NOT end a sentence
#
# Because `: ; - – — * •` were terminators, `if n_span[-1] not in BOUNDARY`
# was false and THE ENTIRE END CHECK WAS SKIPPED. Measured over the corpus:
# 1,568 block-prefix spans in 978 ads (9.6%) are accepted while their block
# continues, and 87 of those in 64 ads drop a qualifier or negation. Real case,
# ad 8da75b8c: "…lokale forhold i Øst-" accepted, "Finnmark er positivt, men
# ikke et krav." dropped — the span asserts the opposite of its sentence, which
# is precisely what the end rule exists to stop. Two of the real cases end on a
# MID-WORD hyphen ("Øst-", "B2-").
#
# The same conflation created false START boundaries. Norwegian suspended
# compounding ("norsk- eller engelskkunnskaper") occurs in 6,350 ads (62.5%),
# 16,685 times; 507 blocks pair it with a language token and 60 of those with a
# negation. With `-` a terminator, the walk-back stopped at the hyphen and
# accepted a span whose sentence began "Vi krever ikke".
#
# Bullets and dashes need no place in either set: they are non-alphanumeric, so
# the furniture walk already steps over them to reach the block edge. Dropping
# them survived all 34 tests in this file, which is why this comment carries the
# measurements rather than the old set carrying the characters.

@pytest.mark.parametrize("cut", [":", ";", "-", "*", "•", "–", "—"])
def test_a_head_ending_on_a_non_terminator_is_still_rejected(cut):
    """Only `.!?` and a block edge may bypass the end check."""
    text = f"Om oss\nGode norskkunnskaper{cut} men det er ikke et krav\nOppstart"
    assert v._span_ok(f"Gode norskkunnskaper{cut}", text) == (False, "ends_mid_sentence")


def test_a_mid_word_hyphen_does_not_terminate_a_span():
    """Ad 8da75b8c, the real case: the accepted span ended "i Øst-" and the
    dropped remainder was "Finnmark er positivt, men ikke et krav."."""
    text = ("Om oss\nKjennskap til samisk språk og lokale forhold i Øst-Finnmark "
            "er positivt, men ikke et krav\nOppstart")
    assert v._span_ok("Kjennskap til samisk språk og lokale forhold i Øst-",
                      text) == (False, "ends_mid_sentence")


def test_a_suspended_hyphen_is_not_a_sentence_start():
    """`\\w- og/eller \\w` occurs in 6,350 ads (62.5%). The previous guard used
    the UNhyphenated "norsk og engelsk" and so never reached this path."""
    text = "Krav\nVi krever ikke norsk- eller engelskkunnskaper for denne jobben\nOppstart"
    assert v._span_ok("eller engelskkunnskaper for denne jobben",
                      text) == (False, "starts_mid_sentence")


def test_a_full_stop_still_terminates_a_span():
    """The other direction: narrowing the end set must not reject real spans.
    116 of 169 real emitted spans carry no terminator and are accepted by the
    block-edge rule; the 53 that do end in `.` must stay accepted mid-block."""
    text = "Om oss\nGode norskkunnskaper. Vi tilbyr opplæring\nOppstart"
    assert v._span_ok("Gode norskkunnskaper.", text) == (True, "")


def test_a_colon_still_opens_a_span():
    """`:` remains a START boundary — "Krav: Gode norskkunnskaper" is one block
    in real ads, and the lead-in is not part of the evidence."""
    text = "Om oss\nKrav: Gode norskkunnskaper muntlig og skriftlig\nOppstart"
    assert v._span_ok("Gode norskkunnskaper muntlig og skriftlig", text) == (True, "")


def test_a_bullet_glyph_still_opens_a_span_via_furniture():
    """Bullets are gone from both terminator sets; the furniture walk must still
    reach the block edge. 1,437 corpus blocks are led by a hyphen, 2,883 by •."""
    for glyph in ("-", "•", "*", "–"):
        text = f"Om oss\n{glyph} Gode norskkunnskaper er et krav\nOppstart"
        assert v._span_ok("Gode norskkunnskaper er et krav", text) == (True, ""), glyph


# ── truncation must not manufacture a boundary ────────────────────────────────
#
# validate() receives `sent_text` from prepare(), which for a long ad is
# body[:HEAD_CHARS] + "\n[...]\n" + body[-TAIL_CHARS:]. 528 ads (5.2%) are
# truncated, and the head cut lands MID-BLOCK in 519 of them. The inserted
# newline then turns an arbitrary mid-sentence cut into a block edge, so a
# sentence-truncated fragment ending exactly at the cut passes every rule.
#
# Measured: 28 of the 528 truncated ads accept such a fragment today. Real case
# bca7d941 accepts "...kommunikasjonsevner på norsk eller et annet skandinavisk
# språk, samt p" — where the cut removed "å engelsk.", the ENGLISH HALF of the
# disjunction. The span supports `either_norwegian_or_english` on evidence whose
# second half was deleted by our own truncation.

def _truncated_head_fixture():
    """A body whose HEAD cut lands INSIDE a language sentence.

    Built arithmetically rather than guessed: prepare() truncates only above
    HEAD_CHARS + TAIL_CHARS (6,000), and the cut must fall mid-sentence for this
    to test anything. The first draft used a 3,446-char body, which is not
    truncated at all — so the test passed while asserting nothing."""
    from finn_smart_search.understanding.census_prompt import HEAD_CHARS, TAIL_CHARS
    sentence = "Gode norskkunnskaper og god engelsk er et krav for denne stillingen"
    filler = "Vi er en stor arbeidsgiver i regionen. " * 200
    pad = HEAD_CHARS - 30 - len("Krav til stillingen\n")
    body = ("Krav til stillingen\n" + filler[:pad] + "\n" + sentence + "\n"
            + "Vi tilbyr gode betingelser. " * 120)
    assert len(body) > HEAD_CHARS + TAIL_CHARS, len(body)
    return body


def test_a_fragment_ending_at_the_truncation_cut_is_rejected():
    body = _truncated_head_fixture()
    sent = v.prepare("t", body, "no")["sent_text"]
    assert "[...]" in sent, "fixture must actually be truncated"
    head = sent[:sent.index("\n[...]")]
    frag = head[head.rfind("\n") + 1:]
    assert len(frag) >= 15 and v.LANG_TOKEN.search(frag), \
        f"fixture must put a language fragment at the cut, got {frag[-50:]!r}"
    ok, why = v._span_ok(frag, sent)
    assert not ok, f"a fragment cut by our own truncation was accepted: {frag[-40:]!r}"
    assert why == "truncated_fragment", why


def test_a_fragment_starting_at_the_truncation_cut_is_rejected():
    from finn_smart_search.understanding.census_prompt import HEAD_CHARS, TAIL_CHARS
    sentence = "norsk og engelsk kreves begge steder i denne stillingen"
    body = ("Krav\n" + "Vi er en stor arbeidsgiver. " * 200 + "\n"
            + "Vi tilbyr gode vilkår. " * 90 + sentence + "\nSlutt")
    assert len(body) > HEAD_CHARS + TAIL_CHARS
    sent = v.prepare("t", body, "no")["sent_text"]
    tail = sent[sent.index("[...]") + 5:].lstrip("\n")
    frag = tail.split("\n")[0]
    if len(frag) >= 15 and v.LANG_TOKEN.search(frag):
        ok, why = v._span_ok(frag, sent)
        assert not ok, f"accepted a fragment starting at the cut: {frag[:40]!r}"
        assert why == "truncated_fragment", why


def test_an_untruncated_ad_is_unaffected():
    """The guard must not cost anything on the 94.8% of ads that are not cut."""
    body = "Krav til stillingen\nGode norskkunnskaper er et krav\nOppstart"
    sent = v.prepare("t", body, "no")["sent_text"]
    assert "[...]" not in sent
    assert v._span_ok("Gode norskkunnskaper er et krav", sent) == (True, "")


# ── non-bokmål evidence (measured) ───────────────────────────────────────────
#
# Every fixture in this file was bokmål. Corpus: 385 ads are doc_lang=en (3.8%),
# 66 mixed, 642 carry nynorsk markers, 144 mention samisk, 106 bergenstest. And
# of the 169 real emitted spans, 4 are English ("Language: Scandinavian or
# English", "Excellent written and spoken English...") and several nynorsk.
#
# Blocks whose ONLY language token is a given alternative: `norwegian` 574,
# `english` 354, `språk` 702, `samisk` 68, `bergenstest` 39, `munnleg`/`skriftleg`
# 33, `bokmål` 8, `scandinavian` 12. Deleting the English half of LANG_TOKEN
# survived all 34 tests in this file — 940 blocks would lose their only token,
# every span quoting them becomes `no_language_token`, and the verdict demotes to
# `unstated`. That is the 55%-demotion mechanism restricted to the English and
# mixed ads. Same class as the `dokument`/`documentation` blind spot in D11.

ENGLISH_AD = ("About the role\n"
              "Excellent written and spoken English is required for this position\n"
              "Norwegian is an advantage but not a requirement")
NYNORSK_AD = ("Om stillinga\n"
              "Du må ha god kunnskap i norsk, munnleg og skriftleg framstilling\n"
              "Oppstart etter avtale")
SAMISK_AD = ("Om stillingen\n"
             "Kunnskap i samisk språk og kultur er et krav for stillingen\n"
             "Oppstart snarest")
BERGENSTEST_AD = ("Krav til stillingen\n"
                  "Bestått Bergenstesten eller tilsvarande dokumentasjon\n"
                  "Oppstart etter avtale")


@pytest.mark.parametrize("span,text", [
    ("Excellent written and spoken English is required for this position", ENGLISH_AD),
    ("Norwegian is an advantage but not a requirement", ENGLISH_AD),
    ("Du må ha god kunnskap i norsk, munnleg og skriftleg framstilling", NYNORSK_AD),
    ("Kunnskap i samisk språk og kultur er et krav for stillingen", SAMISK_AD),
    ("Bestått Bergenstesten eller tilsvarande dokumentasjon", BERGENSTEST_AD),
])
def test_non_bokmal_evidence_is_accepted(span, text):
    ok, why = v._span_ok(span, text)
    assert ok, f"rejected legitimate non-bokmål evidence: {why}"

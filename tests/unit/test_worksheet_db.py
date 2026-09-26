"""The worksheet's db wiring, asserted against the built page.

The labelling runtime is a browser, so these are STATIC checks on the generated
HTML — they cannot prove the page works, only that it cannot be shipped with the
specific defects the capability contract warns about. The contract
(artifact-capabilities, runtime 0.2.60) is explicit about several, and each one
below corresponds to a rule in it:

  * `claude.use("db")` resolves NULL when the view cannot run it — not served,
    not granted, or failed to load, indistinguishable by design. The page must
    branch on null and still work, or a viewer without the grant loses an hour
    of labelling to a blank screen.
  * `window.claude` carries only `use`. Reading `window.claude.db` is never
    promised and must not appear.
  * Subscribe ONCE per query, never inside render — a render-time subscription
    multiplies on every keystroke.
  * Write one doc at a time, only on change, and never on load.

Why a local fallback stays: `db` needs a signed-in organisation member, and the
labels are an hour of irreplaceable human judgement. Losing them because a grant
was refused is the expensive failure.
"""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

HTML = Path(__file__).resolve().parents[2] / "data" / "worksheet" / "worksheet.html"
requires_build = pytest.mark.skipif(
    not HTML.exists(), reason="run scripts/build_worksheet_artifact.py first")


@pytest.fixture(scope="module")
def page():
    return HTML.read_text(encoding="utf-8")


@requires_build
def test_the_page_resolves_db_through_use_not_a_window_member(page):
    assert 'claude.use("db")' in page
    assert not re.search(r"window\.claude\.db\b", page), \
        "window.claude carries only `use`; no .db member is promised"


@requires_build
def test_a_null_db_does_not_break_the_page(page):
    """`use()` resolves null when the view cannot run the capability. The page
    must keep working on localStorage alone."""
    assert re.search(r"if\s*\(\s*!\s*db\s*\)|db\s*(\?\.|===\s*null|==\s*null)", page), \
        "no null branch on the resolved namespace"
    assert "localStorage" in page, "the local fallback must remain"


@requires_build
def test_the_subscription_is_created_once_and_not_in_render(page):
    """Contract: subscribe once per query, never in render."""
    assert page.count("onSnapshot") <= 2, "more than one subscription site"
    render = page[page.index("function render()"):page.index("function go(")]
    assert "onSnapshot" not in render, "subscribing inside render multiplies listeners"
    assert 'claude.use("db")' not in render, "resolving the capability inside render"


@requires_build
def test_writes_happen_on_change_and_never_on_load(page):
    """A write on load would overwrite a real label with an empty one the moment
    the page opens on another device."""
    assert re.search(r"function\s+saveRemote|const\s+saveRemote", page)
    init = page[page.index("async function initDb") if "async function initDb" in page
                else 0:]
    head = init[:init.index("}") + 1] if "}" in init else init
    assert ".set(" not in head, "a write in the initialisation path"


@requires_build
def test_one_document_per_ad_rather_than_one_blob(page):
    """Per-ad docs: the contract says one write at a time per doc, and a single
    blob would make every keystroke rewrite all 28 labels."""
    assert re.search(r'\b(db|DB)\.doc\(\s*[`"\']labels/', page), "expected labels/<n> docs"


@requires_build
def test_the_storage_key_is_versioned(page):
    """Starting from scratch must actually start from scratch: a stale
    localStorage payload under the old key would resurrect abandoned labels."""
    m = re.search(r'const KEY = "([^"]+)"', page)
    assert m and re.search(r"v\d+$", m.group(1)), f"unversioned key: {m and m.group(1)}"


@requires_build
def test_the_page_says_where_labels_are_being_stored(page):
    """The labeller should be able to tell whether their work is only in this
    browser. Silence here is how an hour goes missing."""
    assert re.search(r"saved to this artifact|this browser only|not synced", page, re.I)

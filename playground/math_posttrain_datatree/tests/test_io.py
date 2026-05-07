from __future__ import annotations

from playground.math_posttrain_datatree.core.utils.io import read_jsonl


def test_read_jsonl_repairs_bare_newlines_inside_json_strings(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        (
            '{"source":"data/CoT/gsm_rft.json","instruction":"Clive opens a box full of different colored balls.\n'
            "The box contains 6 blue balls, 4 red balls.\n"
            'How many balls are in the box?","output":"36","problem":"Clive opens a box full of different colored balls.\n'
            "The box contains 6 blue balls, 4 red balls.\n"
            'How many balls are in the box?","solution":"36"}\n'
            '{"source":"data/CoT/gsm_rft.json","instruction":"Clean row","output":"12","problem":"Clean row","solution":"12"}\n'
        ),
        encoding="utf-8",
    )

    rows = read_jsonl(path)

    assert len(rows) == 2
    assert rows[0]["instruction"] == (
        "Clive opens a box full of different colored balls.\n"
        "The box contains 6 blue balls, 4 red balls.\n"
        "How many balls are in the box?"
    )
    assert rows[0]["problem"] == rows[0]["instruction"]
    assert rows[1]["instruction"] == "Clean row"


def test_read_jsonl_repairs_unicode_line_separators_inside_json_strings(tmp_path):
    path = tmp_path / "broken_u2028.jsonl"
    broken_sep = "\u2028"
    path.write_text(
        (
            '{"source":"data/CoT/gsm_rft.json","instruction":"Clive opens a box.'
            + broken_sep
            + 'The box contains 6 blue balls.'
            + broken_sep
            + 'How many balls are in the box?","output":"36","problem":"Clive opens a box.'
            + broken_sep
            + 'The box contains 6 blue balls.'
            + broken_sep
            + 'How many balls are in the box?","solution":"36"}\n'
        ),
        encoding="utf-8",
    )

    rows = read_jsonl(path)

    assert len(rows) == 1
    assert rows[0]["instruction"] == (
        "Clive opens a box.\n"
        "The box contains 6 blue balls.\n"
        "How many balls are in the box?"
    )

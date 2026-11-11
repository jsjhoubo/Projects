import pytest
from ml_eval import Sample, ModelOutput, run_evaluation
from ml_eval.parsing import extract_answer
from ml_eval.scoring import evaluate_batch

@pytest.mark.parametrize("txt,expected", [
    ("ANSWER: A", "A"),
    ("Answer: b", "B"),
    ("ANSWER: C because it satisfies the constraint", "C"),
    ("ANSWER: a - explanation follows", "A"),
])
def test_mc_parsing(txt, expected):
    assert extract_answer(txt, "multiple_choice") == expected

@pytest.mark.parametrize("txt,expected", [
    ("ANSWER: yes", "yes"),
    ("ANSWER: NO", "no"),
    ("ANSWER: true because all criteria hold", "true"),
])
def test_binary_parsing(txt, expected):
    assert extract_answer(txt, "binary") == expected

def test_order_independence():
    samples = [Sample(4,"q1","A"),Sample(9,"q2","B")]
    outputs = [ModelOutput(9,"ANSWER: B"),ModelOutput(4,"ANSWER: A")]
    result = evaluate_batch(samples, outputs)
    assert result.score == 1.0
    assert result.n == 2

def test_incomplete_generation():
    samples = [Sample(2,"q1","A"),Sample(5,"q2","A"),Sample(8,"q3","A")]
    result = evaluate_batch(samples,[ModelOutput(2,"ANSWER: A")])
    assert result.n == 3
    assert result.score == pytest.approx(1/3)
    assert {v["index"] for v in result.per_sample} == {2,5,8}

def test_complete_generation():
    samples = [Sample(2,"q1","A"),Sample(5,"q2","B")]
    result = run_evaluation(samples,lambda _: [ModelOutput(5,"ANSWER: B"),ModelOutput(2,"ANSWER: A")])
    assert result.score == 1.0

def test_empty_input():
    with pytest.raises(ValueError):
        run_evaluation([],lambda _: [])

def f05_entity(pred, true):
    """F0.5 for one S1 entity. pred/true are sets of S2/S3 ids."""
    if not true and not pred:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)

def macro_f05(pred_dict, true_dict):
    """Average over EVERY S1 in true_dict. Missing prediction = empty set."""
    scores = [f05_entity(pred_dict.get(s1, set()), t) for s1, t in true_dict.items()]
    return sum(scores) / len(scores)

if __name__ == "__main__":
    # README example must give 0.714
    print(round(f05_entity({"S2-47", "S2-193", "S3-812"}, {"S2-47", "S3-812"}), 3))
    assert f05_entity(set(), set()) == 1.0
    assert f05_entity({"S2-1"}, set()) == 0.0
    assert f05_entity(set(), {"S2-1"}) == 0.0
    print("scorer OK")
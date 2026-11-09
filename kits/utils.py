import logging

__all__ = [
    "summarize_state_dict_match",
    "print_state_dict_match_report",
]


def summarize_state_dict_match(module, incoming_state_dict):
    model_state = module.state_dict()
    matched_state = {}
    missing_keys = []
    unexpected_keys = []
    shape_mismatched = []

    for key, value in incoming_state_dict.items():
        if key not in model_state:
            unexpected_keys.append(key)
            continue
        if tuple(model_state[key].shape) != tuple(value.shape):
            shape_mismatched.append((key, tuple(model_state[key].shape), tuple(value.shape)))
            continue
        matched_state[key] = value

    for key in model_state.keys():
        if key not in matched_state:
            missing_keys.append(key)

    return matched_state, {
        "total_model_keys": len(model_state),
        "total_incoming_keys": len(incoming_state_dict),
        "matched_keys": len(matched_state),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "shape_mismatched": shape_mismatched,
    }


def print_state_dict_match_report(label, path, report):
    total_model = report["total_model_keys"]
    total_incoming = report["total_incoming_keys"]
    matched = report["matched_keys"]
    missing = report["missing_keys"]
    unexpected = report["unexpected_keys"]
    shape_mismatched = report["shape_mismatched"]

    matched_model_ratio = matched / total_model if total_model else 1.0
    matched_incoming_ratio = matched / total_incoming if total_incoming else 1.0

    lines = [
        (
            f"[Weight Match] {label}: model={matched}/{total_model} ({matched_model_ratio:.2%}), "
            f"incoming={matched}/{total_incoming} ({matched_incoming_ratio:.2%})"
        ),
        f"  source      : {path}",
        (
            f"  mismatch    : missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape={len(shape_mismatched)}"
        ),
    ]

    if missing:
        lines.append(f"  missing[:5] : {missing[:5]}")
    if unexpected:
        lines.append(f"  extra[:5]   : {unexpected[:5]}")
    if shape_mismatched:
        preview = [
            f"{key}: model{model_shape} vs ckpt{ckpt_shape}"
            for key, model_shape, ckpt_shape in shape_mismatched[:3]
        ]
        lines.append(f"  shape[:3]   : {preview}")

    message = "\n".join(lines)
    print(message + "\n")
    logging.info(message)

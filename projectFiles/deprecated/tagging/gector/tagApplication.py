def get_target_sent_by_edits(source_tokens, edits):
    target_tokens = source_tokens[:]
    shift_idx = 0
    for edit in edits:
        start, end, label, _ = edit
        target_pos = start + shift_idx
        source_token = target_tokens[target_pos] \
            if len(target_tokens) > target_pos >= 0 else ''
        if label == "":
            del target_tokens[target_pos]
            shift_idx -= 1
        elif start == end:
            word = label.replace("$APPEND_", "")
            target_tokens[target_pos: target_pos] = [word]
            shift_idx += 1
        elif label.startswith("$TRANSFORM_"):
            word = apply_reverse_transformation(source_token, label)
            if word is None:
                word = source_token
            target_tokens[target_pos] = word
        elif start == end - 1:
            word = label.replace("$REPLACE_", "")
            target_tokens[target_pos] = word
        elif label.startswith("$MERGE_"):
            target_tokens[target_pos + 1: target_pos + 1] = [label]
            shift_idx += 1

    return replace_merge_transforms(target_tokens)


def replace_merge_transforms(tokens):
    if all(not x.startswith("$MERGE_") for x in tokens):
        return tokens

    target_line = " ".join(tokens)
    target_line = target_line.replace(" $MERGE_HYPHEN ", "-")
    target_line = target_line.replace(" $MERGE_SPACE ", "")
    return target_line.split()


def convert_using_case(token, smart_action):
    if not smart_action.startswith("$TRANSFORM_CASE_"):
        return token
    if smart_action.endswith("LOWER"):
        return token.lower()
    elif smart_action.endswith("UPPER"):
        return token.upper()
    elif smart_action.endswith("CAPITAL"):
        return token.capitalize()
    elif smart_action.endswith("CAPITAL_1"):
        return token[0] + token[1:].capitalize()
    elif smart_action.endswith("UPPER_-1"):
        return token[:-1].upper() + token[-1]
    else:
        return token


def convert_using_verb(token, smart_action):
    key_word = "$TRANSFORM_VERB_"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    encoding_part = f"{token}_{smart_action[len(key_word):]}"
    decoded_target_word = decode_verb_form(encoding_part)
    return decoded_target_word


def convert_using_split(token, smart_action):
    key_word = "$TRANSFORM_SPLIT"
    if not smart_action.startswith(key_word):
        raise Exception(f"Unknown action type {smart_action}")
    target_words = token.split("-")
    return " ".join(target_words)


def convert_using_plural(token, smart_action):
    if smart_action.endswith("PLURAL"):
        return token + "s"
    elif smart_action.endswith("SINGULAR"):
        return token[:-1]
    else:
        raise Exception(f"Unknown action type {smart_action}")


def apply_reverse_transformation(source_token, transform):
    if transform.startswith("$TRANSFORM"):
        # deal with equal
        if transform == "$KEEP":
            return source_token
        # deal with case
        if transform.startswith("$TRANSFORM_CASE"):
            return convert_using_case(source_token, transform)
        # deal with verb
        if transform.startswith("$TRANSFORM_VERB"):
            return convert_using_verb(source_token, transform)
        # deal with split
        if transform.startswith("$TRANSFORM_SPLIT"):
            return convert_using_split(source_token, transform)
        # deal with single/plural
        if transform.startswith("$TRANSFORM_AGREEMENT"):
            return convert_using_plural(source_token, transform)
        # raise exception if not find correct type
        raise Exception(f"Unknown action type {transform}")
    else:
        return source_token
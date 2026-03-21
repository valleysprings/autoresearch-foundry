def deduplicate_preserve_order(values):
    result = []
    for value in values:
        seen = False
        for existing in result:
            if existing == value:
                seen = True
                break
        if not seen:
            result.append(value)
    return result

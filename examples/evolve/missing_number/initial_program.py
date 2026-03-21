def missing_number(values):
    upper = len(values) + 1
    for candidate in range(upper):
        present = False
        for value in values:
            if value == candidate:
                present = True
                break
        if not present:
            return candidate
    return None

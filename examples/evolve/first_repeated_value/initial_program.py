def first_repeated_value(values):
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left == right:
                return left
    return None

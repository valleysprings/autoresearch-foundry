def contains_duplicates(values):
    for index, left in enumerate(values):
        for right in values[index + 1 :]:
            if left == right:
                return True
    return False

def has_overlap(left, right):
    for left_value in left:
        for right_value in right:
            if left_value == right_value:
                return True
    return False

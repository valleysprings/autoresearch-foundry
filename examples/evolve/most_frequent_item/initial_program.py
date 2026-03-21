def most_frequent_item(values):
    best_value = None
    best_count = -1
    for value in values:
        count = 0
        for other in values:
            if other == value:
                count += 1
        if count > best_count:
            best_value = value
            best_count = count
    return best_value

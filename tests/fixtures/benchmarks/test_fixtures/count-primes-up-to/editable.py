def count_primes_up_to(limit):
    count = 0
    for candidate in range(2, limit + 1):
        is_prime = True
        divisor = 2
        while divisor < candidate:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            count += 1
    return count

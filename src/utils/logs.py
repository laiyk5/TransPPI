def short_repr(string: str, target_length=10):
    if (len(string) <= target_length):
        return string
    return f'{string[:target_length]}...({len(string) - target_length} left)'
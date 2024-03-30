def extract_pattern(string):
    prefix = "../../"
    if string.startswith(prefix):
        return string[len(prefix):]
    else:
        return string

import re

# Regular expression patterns
ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
credit_card_pattern = r"\b(?:\d[ -]*?){14,16}\b"
email_pattern = r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"
phone_pattern = r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"

# Compiled regular expressions
ssn_matcher = re.compile(ssn_pattern)
credit_card_matcher = re.compile(credit_card_pattern)
email_matcher = re.compile(email_pattern)
phone_matcher = re.compile(phone_pattern)


def replace_match_with_exception(content, exceptions, matcher, replacement):
    def replace_func(match_obj):
        curr_match = match_obj.group(0).lower()
        if curr_match not in exceptions:
            return replacement
        return curr_match  # Return the original match if it's in the exceptions list

    return re.sub(matcher, replace_func, content)


def replace_pii(content):
    if not content:
        return content
    content = replace_match_with_exception(content, {}, phone_matcher, "[phone_number]")
    content = replace_match_with_exception(
        content, {}, email_matcher, "[email_address]"
    )
    content = replace_match_with_exception(content, [], ssn_matcher, "[ssn]")
    content = replace_match_with_exception(
        content, [], credit_card_matcher, "[credit_card_number]"
    )
    return content

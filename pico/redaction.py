"""Shared artifact redaction boundary for runtime traces and offline exports."""


def redact_artifact(value, *, is_secret_name, secret_values, key=None):
    if key and is_secret_name(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            str(item_key): redact_artifact(
                item_value, is_secret_name=is_secret_name, secret_values=secret_values, key=item_key
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            redact_artifact(item, is_secret_name=is_secret_name, secret_values=secret_values, key=key)
            for item in value
        ]
    if isinstance(value, str):
        for secret in sorted(secret_values, key=len, reverse=True):
            if secret:
                value = value.replace(secret, "<redacted>")
    return value

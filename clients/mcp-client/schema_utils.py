def clean_schema(schema):
    """
    Recursively removes 'title' fields from a JSON schema to make it Gemini-compatible.

    Args:
        schema (dict): The schema dictionary.

    Returns:
        dict: Cleaned schema without 'title' fields.
    """
    if isinstance(schema, dict):
        # Remove top-level 'title' if present
        schema.pop("title", None)

        # Clean nested properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for key in schema["properties"]:
                schema["properties"][key] = clean_schema(schema["properties"][key])

    return schema

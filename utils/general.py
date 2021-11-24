def get_markdown(path, prefix="app_content"):
    path = f"{prefix}/{path}"
    with open(path, "r") as f:
        contents = f.read()
    return contents

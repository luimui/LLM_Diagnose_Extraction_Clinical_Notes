def color_tokens_html(tokens: list, labels: list):
    output = []

    color_map = {
        "B-PROBLEM": "red",
        "I-PROBLEM": "purple",
        "O": "black",
    }

    for token, label in zip(tokens, labels):
        if token == "\n":
            output.append("<br>")
        else:
            color = color_map.get(label)
            if color is None:
                raise ValueError("Colouring error")

            output.append(
                f'<span style="color: {color};">{token}</span>'
            )

    return " ".join(output)
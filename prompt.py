
def format_prompt(prompt, text, narratives, labels):

    narrative_block = ""
    for i in range(len(narratives) - 1):
        narrative_block += f"{labels[i]}. {narratives[i]}\n"
    narrative_block += f"{labels[-1]}. {narratives[-1]}\n"

    prompt = prompt.replace('{narratives}', narrative_block)

    prompt = prompt.replace('{text}', text)

    return prompt


from pathlib import Path

def expand_path(p):
    # convert to Path and expand ~, then resolve to absolute
    return Path(str(p)).expanduser().resolve()

def prompt_selection(options, prompt_message="Please make a selection:", allow_quit=True):
    """
    Display a list of options and prompt user to make one selection.

    Args:
        options (list): List of strings to choose from
        prompt_message (str): Custom prompt message
        allow_quit (bool): Whether to allow 'q' to quit

    Returns:
        str: Selected option, or None if user quits
    """
    if not options:
        print("No options provided.")
        return None

    while True:
        print(f"\n{prompt_message}")
        print("-" * len(prompt_message))

        # Display numbered options
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")

        if allow_quit:
            print("q. Quit")

        # Get user input
        choice = input(f"\nEnter your choice (1-{len(options)}" + ("or 'q'" if allow_quit else "") + "): ").strip().lower()

        # Handle quit
        if allow_quit and choice in ['q', 'quit']:
            return None

        # Handle numeric selection
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(options):
                selected = options[choice_num - 1]
                print(f"\nYou selected: {selected}")
                return selected
            else:
                print(f"Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
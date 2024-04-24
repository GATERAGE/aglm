import curses
import requests
import os
import subprocess

def get_api_key():
    env_file = './.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as file:
            return file.read().strip()
    else:
        api_key = input("Enter your OpenAI API key: ").strip()
        with open(env_file, 'w') as file:
            file.write(api_key)
        return api_key

class TerminAI:
    def __init__(self, stdscr, api_key):
        self.stdscr = stdscr
        self.api_key = api_key
        self.setup_folders()

    def setup_folders(self):
        # Set up both terminai and saindbx folders
        os.makedirs('./terminai', exist_ok=True)
        os.makedirs('./saindbx', exist_ok=True)
        os.chmod('./terminai', 0o700)
        os.chmod('./saindbx', 0o700)

    def talk_to_ai(self, message):
        # Simulate talking to an AI (like using SimpleCoder)
        return f"Simulated response for: {message}"

    def execute_and_save_command(self, command):
        # Execute command and handle SimpleCoder interaction
        try:
            # Assuming SimpleCoder could be a method to generate or analyze code
            # This is a placeholder for the actual function call
            # output = SimpleCoder.generate_code(command)
            output = f"Simulated code generation for: {command}"
            filepath = os.path.join('./saindbx', 'output.txt')
            with open(filepath, 'w') as file:
                file.write(output)
            return f"Code saved to saindbx/output.txt\n{output}"
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def main(self):
        self.stdscr.clear()
        self.stdscr.addstr("Welcome to TerminAI. Type 'exit' to quit, or type a command to execute it.\n")
        curses.echo()
        self.stdscr.keypad(True)

        while True:
            self.stdscr.addstr("> ")
            input_str = self.stdscr.getstr().decode('utf-8').strip()

            if input_str == 'exit':
                break
            elif input_str.startswith("cmd:"):
                command = input_str[4:]
                output = self.execute_and_save_command(command)
                self.stdscr.addstr(f"Command executed and result saved.\nOutput:\n{output}\n")
            else:
                response = self.talk_to_ai(input_str)
                self.stdscr.addstr(f"AI: {response}\n")

            self.stdscr.refresh()

        self.stdscr.addstr("Goodbye!")
        self.stdscr.refresh()
        self.stdscr.getch()

if __name__ == '__main__':
    api_key = get_api_key()  # Get the API key before initializing curses
    curses.wrapper(TerminAI, api_key)

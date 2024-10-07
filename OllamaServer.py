import subprocess
import time
import os
import signal
class OllamaServer:
    def __init__(self, command='ollama serve', port=11434):
        self.command = command
        self.process = None
        self.port = port

    def start(self):
        """Start the Ollama server."""
        if self.process is None:
            print("Starting Ollama server...")
            self.process = subprocess.Popen(self.command, shell=True)
            time.sleep(5)  # Wait for the server to start
            print("Ollama server started.")

    def stop(self):
        """Stop the Ollama server."""
        if self.process is not None:
            print("Stopping Ollama server...")
            self.process.terminate()  # Graceful stop
            self.process.wait()  # Wait for the process to terminate
            self.process = None
            print("Ollama server stopped.")

    def kill_by_port(self):
        """Kill the process running on the specified port."""
        try:
            # Find the PID of the process running on the specified port
            result = subprocess.run(
                ["lsof", "-t", f"-i:{self.port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            pids = result.stdout.decode().strip().split()

            if pids:
                for pid in pids:
                    print(f"Killing process with PID: {pid}")
                    os.kill(int(pid), signal.SIGTERM)
                print(f"Processes running on port {self.port} have been terminated.")
            else:
                print(f"No processes found running on port {self.port}.")
        except Exception as e:
            print(f"An error occurred while trying to kill process: {str(e)}")

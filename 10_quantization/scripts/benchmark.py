import argparse
import time
import json
import requests
import psutil
from threading import Thread

# --- Configuration ---
# The default model to benchmark. This can be overridden by a command-line argument.
DEFAULT_MODEL = "my-custom-model:q4"

# The prompt to send to the model for generating tokens.
DEFAULT_PROMPT = "Why is the sky blue?"

# The Ollama API endpoint.
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# --- Helper Functions ---

def get_ollama_server_process():
    """Finds the psutil.Process object for the main Ollama server."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # The main Ollama server process can be identified by its name.
            # This might need adjustment based on the OS.
            # On Windows, it might be 'ollama.exe'. On Linux, 'ollama'.
            if 'ollama' in proc.name().lower() and 'serve' in proc.cmdline():
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_memory(process, results):
    """
    Monitors the memory usage of a process in a separate thread.
    Records the peak memory usage in the results dictionary.
    """
    results['peak_memory_mb'] = 0
    while not results.get('stop_monitoring', False):
        try:
            # Get memory usage in megabytes (MB)
            mem_info = process.memory_info()
            # RSS is a good proxy for memory usage
            current_memory = mem_info.rss / (1024 * 1024)
            if current_memory > results['peak_memory_mb']:
                results['peak_memory_mb'] = current_memory
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)

def run_benchmark(model_name, prompt):
    """
    Runs the benchmark for a given model and prompt.
    Measures startup time, latency, and throughput.
    """
    print(f"--- Benchmarking Model: {model_name} ---")

    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False  # We want the full response at once for this benchmark
    }

    # 1. Memory Usage Benchmark
    ollama_process = get_ollama_server_process()
    if not ollama_process:
        print("Error: Ollama server process not found. Is Ollama running?")
        return None

    memory_results = {}
    memory_thread = Thread(target=monitor_memory, args=(ollama_process, memory_results))
    memory_thread.start()

    # 2. Latency and Throughput Benchmark
    try:
        start_time = time.time()
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        end_time = time.time()

        response_data = response.json()
        
        # Stop the memory monitoring thread
        memory_results['stop_monitoring'] = True
        memory_thread.join()

        # --- Calculate Metrics ---
        # Total time for the request
        total_duration = end_time - start_time
    
        # Generation duration from Ollama's response
        generation_duration_ns = response_data.get('total_duration', 0)
        generation_duration_s = generation_duration_ns / 1.0e9

        # Number of generated tokens
        generated_tokens = response_data.get('eval_count', 0)

        # Throughput (tokens per second)
        throughput = generated_tokens / generation_duration_s if generation_duration_s > 0 else 0
        
        # --- Collect Results ---
        results = {
            "model": model_name,
            "prompt": prompt,
            "total_request_duration_s": total_duration,
            "generation_duration_s": generation_duration_s,
            "generated_tokens": generated_tokens,
            "throughput_tokens_per_sec": throughput,
            "peak_memory_mb": memory_results['peak_memory_mb']
        }
        
        return results

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        # Stop memory monitoring on error
        memory_results['stop_monitoring'] = True
        memory_thread.join()
        return None

def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark an Ollama model.")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The name of the model to benchmark (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"The prompt to use for generation (default: '{DEFAULT_PROMPT}')"
    )
    args = parser.parse_args()

    results = run_benchmark(args.model, args.prompt)

    if results:
        print("\n--- Benchmark Results ---")
        print(json.dumps(results, indent=2))
        print("-------------------------\n")
        
        # --- Save results to files ---
        
        # Latency and Throughput
        with open("../benchmarks/latency.txt", "w") as f:
            f.write(f"Model: {results['model']}\n")
            f.write(f"Throughput (tokens/sec): {results['throughput_tokens_per_sec']:.2f}\n")
            f.write(f"Total Request Duration (s): {results['total_request_duration_s']:.2f}\n")

        # Memory
        with open("../benchmarks/memory.txt", "w") as f:
            f.write(f"Model: {results['model']}\n")
            f.write(f"Peak Memory Usage (MB): {results['peak_memory_mb']:.2f}\n")
            
        # Quality Notes - Add a placeholder
        with open("../benchmarks/quality_notes.md", "a") as f:
            f.write(f"\n## Benchmark for {results['model']}\n")
            f.write(f"*Prompt: \"{results['prompt']}\"*\n")
            f.write("- **Subjective Quality:** (Add observations here)\n")
            f.write("- **Issues Noticed:** (e.g., repetitiveness, factual errors)\n\n")

        print("Results have been saved to the 'benchmarks' directory.")

if __name__ == "__main__":
    main()

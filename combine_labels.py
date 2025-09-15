import json
import os
import glob

def combine_json_files(input_pattern="final_tagged_*.json", output_filename="combined_final_tagged.json"):
    """
    Combines multiple JSON files (each containing a list of dictionaries)
    into a single JSON file.

    Args:
        input_pattern (str): A glob pattern to match input JSON files.
        output_filename (str): The name of the output combined JSON file.
    """
    combined_data = []
    
    # Get a sorted list of files to ensure consistent order
    json_files = sorted(glob.glob(input_pattern))

    if not json_files:
        print(f"No files matching pattern '{input_pattern}' found.")
        return

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combined_data.extend(data)
                else:
                    print(f"Warning: Skipping {file_path} as it does not contain a JSON list.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}. Skipping.")
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")

    if combined_data:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=4)
            print(f"Successfully combined {len(json_files)} files into {output_filename}")
            print(f"Total entries in combined file: {len(combined_data)}")
        except Exception as e:
            print(f"An error occurred while writing to {output_filename}: {e}")
    else:
        print("No data to combine. Output file not created.")

if __name__ == "__main__":
    # You can run this function directly from the command line or call it from another script.
    # Example usage:
    combine_json_files()

    # If you want to specify a different pattern or output filename:
    # combine_json_files(input_pattern="my_data_part_*.json", output_filename="all_my_data.json")

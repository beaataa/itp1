import os
import json

def analyze_labelstudio_json(json_file, output_file):
    """
    Analyze Label Studio JSON and write the results to a file.
    
    Args:
        json_file (str): Path to the Label Studio JSON file.
        output_file (str): Path to the output text file.
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: '{json_file}' is not a valid JSON file.")
        return

    # Open the output file for writing
    with open(output_file, 'w') as out_file:
        out_file.write(f"Analyzing JSON file: {json_file}\n")
        out_file.write(f"Number of tasks in JSON: {len(data)}\n\n")

        if not isinstance(data, list):
            out_file.write("Error: Expected JSON data to be a list.\n")
            return

        for index, task in enumerate(data):
            out_file.write(f"Task {index + 1}:\n")
            
            # Extract task metadata
            task_id = task.get("id")
            image_data = task.get("data", {}).get("image")
            
            if not image_data:
                out_file.write(f"  Warning: No image data found for task {task_id}\n")
                continue

            component = image_data.split("/")[-2] if "/" in image_data else "Unknown"
            image_filename = os.path.basename(image_data)
            
            out_file.write(f"  Task ID: {task_id}\n")
            out_file.write(f"  Component: {component}\n")
            out_file.write(f"  Image: {image_filename}\n")
            
            # Extract annotations
            annotations = task.get("annotations", [])
            for ann in annotations:
                results = ann.get("result", [])
                for result in results:
                    labels = result.get("value", {}).get("brushlabels", [])
                    if labels:
                        out_file.write(f"  Labels: {labels}\n")
            
            out_file.write("\n")  # Add a blank line after each task

# Example usage
if __name__ == "__main__":
    json_input_path = "labelstudio_export.json"  # Replace with your JSON file path
    output_text_path = "output_analysis.txt"     # Replace with your desired output file path
    
    analyze_labelstudio_json(json_input_path, output_text_path)
    print(f"Analysis complete. Results written to '{output_text_path}'.")

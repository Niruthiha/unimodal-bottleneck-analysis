import os
import json
import base64
import time
from openai import OpenAI
from tqdm import tqdm

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

MODEL_NAME = "gpt-4o"
client = OpenAI(api_key=API_KEY)

# --- FILE PATHS ---
EXPLANATION_FILE = "explanation_analysis_data.json"
VALIDATION_LOG_FILE = "validation_log.txt" # The detailed log from your last run
ORIGINAL_DATASET_FILE = os.path.join("data", "train.jsonl")
IMAGE_DIR = os.path.join("data", "img")

# --- HELPER FUNCTIONS ---

def encode_image(image_path):
    """Encodes an image file into base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return None

def load_json_data(file_path):
    """Loads data from a standard .json file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the file {file_path}.")
        return None
    return None

def create_id_to_image_map(original_dataset_path):
    """Creates a dictionary to map meme IDs to their image filenames."""
    id_map = {}
    try:
        with open(original_dataset_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    id_map[item['id']] = item['img']
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the original dataset at {original_dataset_path}.")
        return None
    return id_map

def get_visual_override_ids(log_file_path):
    """Reads the validation log to get IDs of visual overrides."""
    visual_ids = set()
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                if "Trigger: Visual" in line:
                    try:
                        # Assumes format "ID: 12345, Trigger: Visual"
                        meme_id = int(line.split(',')[0].split(':')[1].strip())
                        visual_ids.add(meme_id)
                    except (IndexError, ValueError):
                        continue
    except FileNotFoundError:
        print(f"FATAL ERROR: Could not find the validation log file at {log_file_path}.")
        return None
    return visual_ids

def get_image_description(image_path):
    """
    Runs a neutral probe to get a description of the image content.
    """
    base64_image = encode_image(image_path)
    if not base64_image:
        return "Error: Image file not found."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an AI assistant. Describe the visual elements in this image objectively."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"API Error during description probe: {e}"

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

def main():
    """
    Gets objective descriptions for all images that previously caused a visual override.
    """
    print("======================================================")
    print("--- Getting Descriptions for Visual Override Images ---")
    print("======================================================")

    # 1. Get the list of IDs that were visual overrides
    visual_override_ids = get_visual_override_ids(VALIDATION_LOG_FILE)
    if visual_override_ids is None:
        return
    print(f"Found {len(visual_override_ids)} visual override cases to describe.")

    # 2. Create the ID-to-Image mapping
    id_to_image_map = create_id_to_image_map(ORIGINAL_DATASET_FILE)
    if not id_to_image_map:
        return

    # 3. Run the description probe for each visual override case
    description_results = []
    print("\nRequesting image descriptions from the model...")
    for meme_id in tqdm(visual_override_ids, desc="Describing Images"):
        image_filename = id_to_image_map.get(meme_id)
        if not image_filename:
            continue

        image_path = os.path.join(IMAGE_DIR, image_filename.replace('img/', ''))
        description = get_image_description(image_path)

        description_results.append({
            "id": meme_id,
            "image_filename": image_filename,
            "llm_description": description
        })
        time.sleep(1) # Rate limit

    # 4. Save the results to a new file
    output_file = "visual_override_descriptions.json"
    with open(output_file, 'w') as f:
        json.dump(description_results, f, indent=2)

    print("\n======================================================")
    print("--- Description Collection Finished ---")
    print(f"💾 Descriptions for {len(description_results)} images saved to: {output_file}")
    print("======================================================")


if __name__ == "__main__":
    main()

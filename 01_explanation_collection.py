import os
import json
import base64
import time
from openai import OpenAI
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ==============================================================================
# --- CONFIGURATION (MUST BE AT THE TOP) ---
# This block defines all the global variables that the functions below will use.
# ==============================================================================
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# --- FILE PATHS ---
DATA_DIR = "data"
JSONL_FILE = os.path.join(DATA_DIR, "train.jsonl")  # Use training data which has labels
IMAGE_DIR = os.path.join(DATA_DIR, "img")

# --- EXPERIMENT SETTINGS ---
MAX_SAMPLES = 500  # Larger sample for meaningful metrics
MODEL_NAME = "gpt-4o"
RATE_LIMIT_DELAY_SECONDS = 2
# ==============================================================================


# --- HELPER FUNCTIONS ---

def encode_image(image_path):
    """Encodes an image file into base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Warning: Image file not found at {image_path}")
        return None

def load_hmc_data(jsonl_path):
    """Loads data from a .jsonl file."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON line: {line[:50]}...")
                    continue
    return data

def get_system_prompt():
    """Returns the structured system prompt for the model."""
    return """
    You are an expert AI assistant specializing in multimodal content analysis for safety. Your task is to identify hateful memes. A meme is considered hateful if it attacks or demeans a protected group based on attributes like race, religion, ethnic origin, gender, disability, or sexual orientation.

    You will be given a meme containing an image and overlaid text. Follow these steps for your analysis:

    1.  **Image Analysis:** Briefly describe the key elements, people, and actions in the image.
    2.  **Text Analysis:** State the literal meaning of the text.
    3.  **Multimodal Reasoning:** Explain how the text and image interact. Analyze any cultural context, symbolism, or stereotypes at play. State clearly whether their combination constitutes a hateful message.
    4.  **Confidence Score:** Provide your confidence in the final classification as a percentage (e.g., "Confidence: 95%").
    5.  **Final Classification:** Conclude your entire response with a single, definitive word on a new line: `Hateful` or `Not Hateful`.
    """

def parse_model_response(response_text):
    """Parses the model's text response to extract classification and confidence."""
    response_lower = response_text.lower().strip()
    lines = response_lower.split('\n')

    # Check for content policy refusal first
    if "sorry" in response_lower and ("can't assist" in response_lower or "cannot assist" in response_lower):
        # If model refuses, assume it detected harmful content
        return 1, 0.9  # Classify as hateful with high confidence
    
    # Get Classification - improved parsing
    prediction = -1
    
    # Check last line first (most reliable)
    last_line = lines[-1].strip()
    if "not hateful" in last_line:
        prediction = 0
    elif "hateful" in last_line:
        prediction = 1
    else:
        # Fallback: check all lines from bottom up
        for line in reversed(lines):
            line = line.strip()
            if "not hateful" in line and "hateful" not in line.replace("not hateful", ""):
                prediction = 0
                break
            elif "hateful" in line and "not hateful" not in line:
                prediction = 1
                break
    
    # If still no prediction found, check for keywords
    if prediction == -1:
        response_clean = response_text.lower()
        if "this meme is hateful" in response_clean or "classification: hateful" in response_clean:
            prediction = 1
        elif "this meme is not hateful" in response_clean or "classification: not hateful" in response_clean:
            prediction = 0

    # Get Confidence Score - improved extraction
    confidence = 0.5  # Default
    for line in lines:
        if "confidence" in line:
            try:
                # Look for percentage patterns
                import re
                match = re.search(r'(\d+)%', line)
                if match:
                    confidence = int(match.group(1)) / 100.0
                    break
                # Look for decimal patterns
                match = re.search(r'confidence[:\s]*(\d*\.?\d+)', line)
                if match:
                    conf_val = float(match.group(1))
                    confidence = conf_val if conf_val <= 1 else conf_val / 100.0
                    break
            except (ValueError, AttributeError):
                continue
    
    return prediction, confidence


# --- MAIN EXPERIMENT SCRIPT ---

def main():
    print("--- Starting Hateful Memes Classification Experiment ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {JSONL_FILE}")

    client = OpenAI(api_key=API_KEY)

    print("Loading dataset...")
    dataset = load_hmc_data(JSONL_FILE)
    if MAX_SAMPLES:
        print(f"Running on a subset of {MAX_SAMPLES} samples.")
        dataset = dataset[:MAX_SAMPLES]
    else:
        print(f"Running on the full dataset of {len(dataset)} samples.")

    y_true = []
    y_pred = []
    y_scores = []
    failed_predictions = 0
    predictions_without_labels = []

    system_prompt = get_system_prompt()

    print("\nProcessing samples...")
    for item in tqdm(dataset, desc="Analyzing Memes"):
        true_label = item.get('label')
        text_content = item['text']
        
        # Fix image path - remove 'img/' prefix if it exists
        img_filename = item['img']
        if img_filename.startswith('img/'):
            img_filename = img_filename[4:]  # Remove 'img/' prefix
        image_file_path = os.path.join(IMAGE_DIR, img_filename)

        base64_image = encode_image(image_file_path)
        if not base64_image:
            failed_predictions += 1
            continue

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Meme Text: \"{text_content}\""},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                max_tokens=300
            )
            response_text = response.choices[0].message.content
            prediction, confidence = parse_model_response(response_text)

            # Debug: Print first few responses to check format (only for small runs)
            if MAX_SAMPLES <= 20 and (len(y_true) + len(predictions_without_labels)) < 3:
                print(f"\n--- Debug Response for ID {item['id']} ---")
                print(f"Response: {response_text}")
                print(f"Parsed - Prediction: {prediction}, Confidence: {confidence}")
                print("--- End Debug ---\n")

            if true_label is not None:
                if prediction != -1:
                    y_true.append(true_label)
                    y_pred.append(prediction)
                    y_scores.append(confidence)
                    
                    # Track content policy refusals
                    if "sorry" in response_text.lower() and "can't assist" in response_text.lower():
                        print(f"\n🚫 Content policy refusal for ID {item['id']} - classified as hateful")
                else:
                    failed_predictions += 1
                    print(f"\nWarning: Failed to parse response for item ID {item['id']}. Response: {response_text}")
            else:
                 predictions_without_labels.append({'id': item['id'], 'prediction': prediction, 'confidence': confidence})

        except Exception as e:
            failed_predictions += 1
            print(f"\nError processing item ID {item['id']}: {e}")

        time.sleep(RATE_LIMIT_DELAY_SECONDS)

    print("\n--- Experiment Finished ---")

    if y_true:
        print("\n--- Performance Metrics (for labeled data) ---")
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            auroc = f"Could not be calculated: {e}"
        cm = confusion_matrix(y_true, y_pred)
        print(f"Total Labeled Samples Processed: {len(y_true)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUROC: {auroc if isinstance(auroc, str) else f'{auroc:.4f}'}")
        print("-" * 25)
        print("Confusion Matrix:")
        print("         Pred Non-Hateful | Pred Hateful")
        print(f"True Non-Hateful: {cm[0][0]:<15} | {cm[0][1]:<12}")
        print(f"   True Hateful: {cm[1][0]:<15} | {cm[1][1]:<12}")
        print("-" * 25)

    if predictions_without_labels:
        print(f"\n--- Predictions (for {len(predictions_without_labels)} unlabeled samples) ---")
        hateful_count = sum(1 for r in predictions_without_labels if r['prediction'] == 1)
        not_hateful_count = len(predictions_without_labels) - hateful_count
        
        print(f"Summary: {hateful_count} Hateful, {not_hateful_count} Not Hateful")
        print(f"Hateful Rate: {hateful_count/len(predictions_without_labels)*100:.1f}%")
        
        print("\nSample predictions:")
        for result in predictions_without_labels[:10]:  # Show more samples
            print(f"ID: {result['id']}, Prediction: {'Hateful' if result['prediction'] == 1 else 'Not Hateful'}, Confidence: {result['confidence']:.2f}")
        
        if len(predictions_without_labels) > 10:
            print(f"... and {len(predictions_without_labels) - 10} more samples")

    print(f"\nTotal Failed Predictions/Errors: {failed_predictions}")
    
    # Save results to file
    results_file = f"gpt4o_results_{len(dataset)}_samples.json"
    results_data = {
        "experiment_info": {
            "model": MODEL_NAME,
            "dataset": JSONL_FILE,
            "total_samples": len(dataset),
            "successful_predictions": len(predictions_without_labels) if predictions_without_labels else len(y_true),
            "failed_predictions": failed_predictions
        },
        "predictions": predictions_without_labels if predictions_without_labels else [
            {"id": i, "true_label": t, "prediction": p, "confidence": s} 
            for i, (t, p, s) in enumerate(zip(y_true, y_pred, y_scores))
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()

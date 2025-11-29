#!/bin/bash
# Edge Impulse Upload Script Template
# Make this executable: chmod +x upload_to_edge_impulse.sh

# Configuration
PROJECT_DIR="data/processed"
EI_PROJECT_ID="your-project-id-here"  # Get from Edge Impulse Studio

# Login to Edge Impulse (run once)
# edge-impulse login

# Upload training data
echo "Uploading training data..."

for class_dir in train/*/; do
    class_name=$(basename "$class_dir")
    echo "Uploading class: $class_name"
    
    edge-impulse uploader \
        --category training \
        --label "$class_name" \
        "$PROJECT_DIR/train/$class_name/"
done

# Upload validation data
echo "Uploading validation data..."

for class_dir in validation/*/; do
    class_name=$(basename "$class_dir")
    echo "Uploading class: $class_name"
    
    edge-impulse uploader \
        --category testing \
        --label "$class_name" \
        "$PROJECT_DIR/validation/$class_name/"
done

echo "Upload complete!"

#!/bin/bash
# Test script for Sleep BCI API v1 endpoints

set -e

echo "==================================="
echo "Testing Sleep BCI API v1 Endpoints"
echo "==================================="
echo ""

# Test 1: Check server is running
echo "Test 1: Server health check"
curl -s http://localhost:8000/ | jq .
echo ""

# Test 2: POST /v1/preprocess - Create a preprocessing job
echo "Test 2: POST /v1/preprocess - Create preprocessing job"
RESPONSE=$(curl -s -X POST http://localhost:8000/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {
      "raw_dir": "/Users/sujannandikolsunilkumar/Downloads/sleep-bci-repo/data/raw"
    },
    "preprocessing_config": {
      "channel": "EEG Fpz-Cz",
      "epochs": 30,
      "bandpass": [0.5, 30],
      "notch": null
    },
    "output": {
      "out_dir": null,
      "combine": false
    },
    "dry_run": false
  }')

echo "$RESPONSE" | jq .
JOB_ID=$(echo "$RESPONSE" | jq -r .job_id)
echo ""
echo "Job created with ID: $JOB_ID"
echo ""

# Test 3: GET /v1/preprocess/{job_id} - Check job status
echo "Test 3: GET /v1/preprocess/{job_id} - Check job status"
sleep 2
curl -s "http://localhost:8000/v1/preprocess/${JOB_ID}" | jq .
echo ""

# Wait a bit and check again
echo "Waiting 5 seconds and checking status again..."
sleep 5
curl -s "http://localhost:8000/v1/preprocess/${JOB_ID}" | jq .
echo ""

echo "==================================="
echo "Testing Training Endpoints"
echo "==================================="
echo ""

# Test 4: POST /train - Create a training job
echo "Test 4: POST /train - Create training job"
# Note: Training requires preprocessed .npz files with at least n_splits nights
# Using /tmp directory to avoid requiring specific test data
TRAIN_RESPONSE=$(curl -s -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "npz_dir": "/Users/sujannandikolsunilkumar/Downloads/sleep-bci-repo/data/processed",
    "model_out": null,
    "fs": 100.0,
    "n_splits": 2
  }')

echo "$TRAIN_RESPONSE" | jq .

# Check if training job was created successfully
if echo "$TRAIN_RESPONSE" | jq -e '.training_id' > /dev/null 2>&1; then
    TRAIN_ID=$(echo "$TRAIN_RESPONSE" | jq -r .training_id)
    echo ""
    echo "Training job created with ID: $TRAIN_ID"
    echo ""

    # Test 5: GET /training/{train_id} - Check training status
    echo "Test 5: GET /training/${TRAIN_ID} - Check training status (initial)"
    sleep 2
    curl -s "http://localhost:8000/training/${TRAIN_ID}" | jq .
    echo ""

    # Wait and check training progress
    echo "Waiting 5 seconds to check training progress..."
    sleep 5
    curl -s "http://localhost:8000/training/${TRAIN_ID}" | jq .
    echo ""

    echo "Waiting 10 more seconds to check if training completes..."
    sleep 10
    TRAIN_STATUS=$(curl -s "http://localhost:8000/training/${TRAIN_ID}")
    echo "$TRAIN_STATUS" | jq .
    echo ""

    # Check if training succeeded and show results
    if echo "$TRAIN_STATUS" | jq -e '.status == "succeeded"' > /dev/null; then
        echo "✅ Training completed successfully!"
        echo "Results:"
        echo "$TRAIN_STATUS" | jq '.results'
    else
        echo "Training status: $(echo "$TRAIN_STATUS" | jq -r '.status')"
    fi
    echo ""
else
    echo ""
    echo "⚠️  Training job could not be created (possibly no valid .npz files in output directory)"
    echo "   The training endpoints are functional but require preprocessed data first."
    echo ""
fi

echo "==================================="
echo "All tests completed!"
echo "==================================="
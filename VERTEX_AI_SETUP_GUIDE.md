# Google Cloud Vertex AI Setup Guide

## Current Issue
The video analysis feature is failing because the Gemini models are not accessible in your Google Cloud project. Here's how to fix it:

## Step 1: Check Available Models
First, let's see what models are actually available in your project:

```bash
# List available models
gcloud ai models list --region=us-central1

# Or check specifically for Gemini models
gcloud ai models list --region=us-central1 --filter="displayName:gemini"
```

## Step 2: Enable Required APIs
Make sure these APIs are enabled:

```bash
# Enable Vertex AI API
gcloud services enable aiplatform.googleapis.com

# Enable Generative AI API (if using newer models)
gcloud services enable generativelanguage.googleapis.com

# Check which APIs are enabled
gcloud services list --enabled | grep -E "(aiplatform|generative)"
```

## Step 3: Check Authentication
Verify your authentication is working:

```bash
# Check current authentication
gcloud auth list

# Check if application default credentials are set
gcloud auth application-default print-access-token

# If needed, set application default credentials
gcloud auth application-default login
```

## Step 4: Verify Project and Billing
```bash
# Check current project
gcloud config get-value project

# Switch to your project if needed
gcloud config set project elite-thunder-461308-f7

# Check if billing is enabled (required for Vertex AI)
gcloud billing accounts list
gcloud billing projects describe elite-thunder-461308-f7
```

## Step 5: Test Model Access
Try accessing models directly:

```bash
# Test with curl (replace with your access token)
ACCESS_TOKEN=$(gcloud auth application-default print-access-token)

curl -X POST \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/elite-thunder-461308-f7/locations/us-central1/publishers/google/models/gemini-pro:predict \
  -d '{"instances": [{"content": "Hello"}]}'
```

## Step 6: Alternative Model Names to Try

Update the video processor to try these models in order:

1. `gemini-pro` (basic text model)
2. `gemini-pro-vision` (if available)
3. `gemini-1.5-pro-preview-0409` (if preview access)
4. `text-bison` (fallback text model)

## Step 7: Check Quotas and Limits
```bash
# Check quotas
gcloud compute project-info describe --project=elite-thunder-461308-f7
```

## Common Issues and Solutions

### Issue: "Model not found"
- **Solution**: The model name might be different in your region
- **Try**: Use `gcloud ai models list` to see available models
- **Alternative**: Use a different model name

### Issue: "Authentication failed"
- **Solution**: Re-run `gcloud auth application-default login`
- **Check**: Make sure the service account has Vertex AI permissions

### Issue: "Quota exceeded"
- **Solution**: Check billing and quota limits in Google Cloud Console
- **Upgrade**: May need to upgrade billing account

### Issue: "API not enabled"
- **Solution**: Enable Vertex AI API as shown in Step 2

## Testing the Fix

Once you've made changes, test with this simple script:

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize
vertexai.init(project="elite-thunder-461308-f7", location="us-central1")

# Test different models
models_to_try = [
    "gemini-pro",
    "gemini-pro-vision", 
    "text-bison"
]

for model_name in models_to_try:
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content("Hello, this is a test")
        print(f"✅ {model_name}: {response.text[:100]}")
        break
    except Exception as e:
        print(f"❌ {model_name}: {e}")
```

## Final Notes

- Video frame extraction is working perfectly
- Audio extraction is working perfectly  
- Only the AI vision analysis needs Vertex AI access
- The system will provide detailed technical information even without vision analysis
- Once Vertex AI is configured, re-upload your video to get full analysis
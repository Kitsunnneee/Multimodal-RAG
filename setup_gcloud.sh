#!/bin/bash

echo "🔧 Google Cloud SDK Setup for Video Analysis"
echo "============================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "📦 Installing Google Cloud SDK..."
    
    # Install using Homebrew (recommended for macOS)
    if command -v brew &> /dev/null; then
        echo "Using Homebrew to install gcloud..."
        brew install --cask google-cloud-sdk
    else
        echo "❌ Homebrew not found. Please install it first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo ""
        echo "Or install Google Cloud SDK manually:"
        echo "   https://cloud.google.com/sdk/docs/install-sdk"
        exit 1
    fi
else
    echo "✅ Google Cloud SDK is already installed"
fi

# Initialize gcloud if not already done
echo ""
echo "🔑 Checking authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "🔐 Setting up authentication..."
    echo "This will open a browser window for authentication."
    read -p "Press Enter to continue..."
    gcloud auth login
    gcloud auth application-default login
else
    echo "✅ Already authenticated"
fi

# Set the project
echo ""
echo "🏗️ Setting up project..."
gcloud config set project elite-thunder-461308-f7

# Enable required APIs
echo ""
echo "🔧 Enabling required APIs..."
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com

# Check available models
echo ""
echo "📋 Checking available Vertex AI models..."
echo "This may take a moment..."

if gcloud ai models list --region=us-central1 --limit=5 > /dev/null 2>&1; then
    echo "✅ Vertex AI API is accessible"
    echo ""
    echo "Available models (showing first 10):"
    gcloud ai models list --region=us-central1 --limit=10 --format="table(name,displayName)"
else
    echo "❌ Cannot access Vertex AI models. Possible issues:"
    echo "   - Billing not enabled"
    echo "   - Vertex AI API not properly enabled"
    echo "   - Insufficient permissions"
fi

echo ""
echo "🧪 Testing Vertex AI access with Python..."
python3 -c "
import vertexai
from vertexai.generative_models import GenerativeModel

try:
    vertexai.init(project='elite-thunder-461308-f7', location='us-central1')
    
    models_to_try = ['gemini-pro', 'gemini-1.5-flash', 'text-bison@001']
    
    for model_name in models_to_try:
        try:
            model = GenerativeModel(model_name)
            print(f'✅ {model_name}: Model accessible')
            # Don't actually call it to avoid charges, just check accessibility
            break
        except Exception as e:
            print(f'❌ {model_name}: {str(e)[:100]}')
    
except Exception as e:
    print(f'❌ Vertex AI initialization failed: {e}')
"

echo ""
echo "🎯 Next Steps:"
echo "1. If models are accessible, restart your Streamlit app"
echo "2. If not accessible, check the VERTEX_AI_SETUP_GUIDE.md file"
echo "3. Make sure billing is enabled in Google Cloud Console"
echo "4. Try uploading your video again"

echo ""
echo "✅ Setup complete! Check the output above for any issues."
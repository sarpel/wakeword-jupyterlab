#!/usr/bin/env python3
"""
Simple test for Gradio app with existing CUDA environment
"""

import torch
import gradio as gr
import sys

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")

    # Check PyTorch and CUDA
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ CUDA not available!")
        return False

    # Check other required packages
    required_packages = [
        'librosa', 'soundfile', 'sklearn',
        'matplotlib', 'seaborn', 'pandas', 'plotly', 'tqdm'
    ]

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: OK")
        except ImportError:
            print(f"❌ {package}: Missing")
            return False

    print("✅ All required packages are available!")
    return True

def create_simple_interface():
    """Create a simple test interface"""
    def test_cuda():
        if torch.cuda.is_available():
            return f"""
✅ CUDA Test Results:
- GPU Available: Yes
- GPU Name: {torch.cuda.get_device_name(0)}
- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB
- CUDA Version: {torch.version.cuda}
- PyTorch CUDA: {torch.__version__}
            """
        else:
            return "❌ CUDA not available!"

    with gr.Blocks(title="CUDA Test") as demo:
        gr.Markdown("# 🎯 CUDA Environment Test")
        gr.Markdown("This is a simple test to verify CUDA support for the Gradio app")

        with gr.Row():
            test_btn = gr.Button("🧪 Test CUDA", variant="primary")

        results = gr.Textbox(label="Test Results", interactive=False, lines=10)

        test_btn.click(test_cuda, outputs=[results])

        gr.Markdown("## Next Steps")
        gr.Markdown("""
        If CUDA is working properly, you can run the full Gradio app:
        ```bash
        python wakeword_training_gradio.py
        ```
        """)

    return demo

def main():
    print("🎯 Simple Gradio CUDA Test")
    print("=" * 40)

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("Please install missing packages:")
        print("pip install gradio librosa soundfile scikit-learn matplotlib seaborn pandas plotly tqdm")
        return

    print("\n✅ Environment check passed!")

    # Create and launch simple interface
    demo = create_simple_interface()

    print("\n🚀 Launching simple test interface...")
    print("Open your browser to test CUDA functionality")

    try:
        demo.launch(share=True, debug=True)
    except KeyboardInterrupt:
        print("\n👋 Test stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
import gradio as gr
import time

def get_status():
    return f"Current time: {time.strftime('%H:%M:%S')}"

# Test the current API syntax
def test_gradio_every():
    with gr.Blocks() as demo:
        status_box = gr.Textbox(label="Status")
        refresh_btn = gr.Button("Refresh")

        # This is the problematic line from wakeword_app.py
        try:
            refresh_btn.click(
                get_status,
                outputs=status_box,
                every=2  # This might cause the error
            )
            print("✅ Current syntax works!")
        except TypeError as e:
            print(f"❌ Current syntax failed: {e}")

            # Try alternative approaches
            try:
                # Alternative 1: Use gr.Timer
                timer = gr.Timer(2.0)
                timer.tick(get_status, outputs=status_box)
                print("✅ Alternative 1 (gr.Timer) works!")
            except Exception as e2:
                print(f"❌ Alternative 1 failed: {e2}")

                try:
                    # Alternative 2: Simple click without every
                    refresh_btn.click(get_status, outputs=status_box)
                    print("✅ Alternative 2 (simple click) works!")
                except Exception as e3:
                    print(f"❌ Alternative 2 failed: {e3}")

if __name__ == "__main__":
    print(f"Testing Gradio API with version: {gr.__version__}")
    test_gradio_every()

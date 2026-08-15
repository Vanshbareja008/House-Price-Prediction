import os
import gradio as gr
import joblib
import pandas as pd

# ==========================================
# Load trained model
# ==========================================

model = joblib.load("house_price_prediction_model.pkl")


# ==========================================
# Prediction Function (Logic Preserved)
# ==========================================

def predict_price(
    area,
    bedrooms,
    bathrooms,
    floors,
    year_built,
    location,
    condition,
    garage
):
    try:
        # Create input DataFrame
        # The model expects ENCODED numerical values
        input_data = pd.DataFrame([{
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "YearBuilt": year_built,
            "Location": location,
            "Condition": condition,
            "Garage": garage
        }])

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        return f"₹ {predicted_price:,.2f}"

    except Exception as e:
        return f"Error: {str(e)}"


# ==========================================
# Custom CSS (Modern Aesthetic)
# ==========================================

custom_css = """
.gradio-container {
    background: linear-gradient(180deg, #edf3f8 0%, #f7fafc 100%);
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
}
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem 1rem;
    max-width: 800px;
    margin: 0 auto;
}
.hero-header h1 {
    font-size: 2.5rem;
    font-weight: 600;
    color: #1a202c;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}
.hero-header h1 span {
    font-family: serif;
    font-style: italic;
    font-weight: 400;
    color: #2b6cb0;
}
.hero-header p {
    color: #4a5568;
    font-size: 1.1rem;
}
.card-container {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05);
}
.result-box textarea {
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #2b6cb0 !important;
    text-align: center !important;
    background: #f7fafc !important;
    border-radius: 12px !important;
}
.predict-btn {
    background: #1a202c !important;
    color: #ffffff !important;
    border-radius: 30px !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.2s ease !important;
}
.predict-btn:hover {
    background: #2d3748 !important;
    transform: translateY(-1px);
}
"""

# ==========================================
# Gradio Interface
# ==========================================

with gr.Blocks(
    css=custom_css,
    title="Data-Driven Property Insights",
    theme=gr.themes.Soft(
        primary_hue="slate",
        neutral_hue="slate",
        radius_size="md"
    )
) as demo:

    # Hero Header Section
    gr.HTML(
        """
        <div class="hero-header">
            <h1>Discover high-growth <span>property values</span></h1>
            <p>Predict real estate market pricing through data-driven ML insights</p>
        </div>
        """
    )

    with gr.Column(elem_classes=["card-container"]):
        with gr.Row():

            # Left Column: Structure Specs
            with gr.Column():
                gr.Markdown("### 📐 Property Specs")
                area = gr.Number(
                    label="Area (sq ft)",
                    value=2500,
                    minimum=1
                )
                bedrooms = gr.Number(
                    label="Bedrooms",
                    value=4,
                    minimum=1,
                    precision=0
                )
                bathrooms = gr.Number(
                    label="Bathrooms",
                    value=3,
                    minimum=1,
                    precision=0
                )
                floors = gr.Number(
                    label="Floors",
                    value=2,
                    minimum=1,
                    precision=0
                )

            # Right Column: Attribute Specs
            with gr.Column():
                gr.Markdown("### 🏙️ Location & Features")
                year_built = gr.Number(
                    label="Year Built",
                    value=2018,
                    minimum=1800,
                    maximum=2026,
                    precision=0
                )
                location = gr.Dropdown(
                    choices=[
                        ("Downtown", 0),
                        ("Suburban", 1),
                        ("Urban", 2)
                    ],
                    value=1,
                    label="Location Zone"
                )
                condition = gr.Dropdown(
                    choices=[
                        ("Excellent", 0),
                        ("Fair", 1),
                        ("Good", 2)
                    ],
                    value=2,
                    label="Condition Rating"
                )
                garage = gr.Dropdown(
                    choices=[
                        ("No", 0),
                        ("Yes", 1)
                    ],
                    value=1,
                    label="Garage Availability"
                )

        # Output Section
        with gr.Row():
            with gr.Column(scale=1):
                predict_button = gr.Button(
                    "Calculate Estimated Value",
                    variant="primary",
                    elem_classes=["predict-btn"]
                )

        with gr.Row():
            with gr.Column(scale=1):
                result = gr.Textbox(
                    label="Estimated Valuation",
                    interactive=False,
                    elem_classes=["result-box"]
                )

    # Button Action (Identical Binding)
    predict_button.click(
        fn=predict_price,
        inputs=[
            area,
            bedrooms,
            bathrooms,
            floors,
            year_built,
            location,
            condition,
            garage
        ],
        outputs=result
    )

# ==========================================
# Run Application
# ==========================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", 7860)
        )
    )

import gradio as gr
import joblib
import pandas as pd

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "house_price_prediction_model.pkl"

model = joblib.load(MODEL_PATH)


# ==========================================
# LABEL ENCODING USED DURING TRAINING
# ==========================================
# IMPORTANT:
# These mappings must match the encoding used
# while training the model.

location_map = {
    "Downtown": 0,
    "Rural": 1,
    "Suburban": 2,
    "Urban": 3
}

condition_map = {
    "Excellent": 0,
    "Fair": 1,
    "Good": 2
}

garage_map = {
    "No": 0,
    "Yes": 1
}


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_house_price(
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
        # Convert categorical values to encoded values
        location_encoded = location_map[location]
        condition_encoded = condition_map[condition]
        garage_encoded = garage_map[garage]

        # Create dataframe in EXACT training order
        sample = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Floors": [floors],
            "YearBuilt": [year_built],
            "Location": [location_encoded],
            "Condition": [condition_encoded],
            "Garage": [garage_encoded]
        })

        # Prediction
        predicted_price = model.predict(sample)[0]

        return f"${predicted_price:,.2f}"

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# ==========================================
# CUSTOM CSS
# ==========================================

css = """
body {
    background: #f5f5f5;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.header {
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    background: #1f2937;
    color: white;
    margin-bottom: 20px;
}

.header h1 {
    font-size: 34px;
    margin-bottom: 8px;
}

.header p {
    font-size: 16px;
    opacity: 0.85;
}

.card {
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}

.result-box {
    font-size: 28px !important;
    font-weight: bold !important;
    text-align: center !important;
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: #666;
    font-size: 14px;
}
"""


# ==========================================
# GRADIO INTERFACE
# ==========================================

with gr.Blocks(
    title="House Price Prediction",
    css=css,
    theme=gr.themes.Soft()
) as app:

    gr.HTML("""
    <div class="header">
        <h1>🏠 House Price Prediction</h1>
        <p>AI-powered house price estimation using XGBoost</p>
    </div>
    """)

    with gr.Row():

        # --------------------------------------
        # LEFT SIDE - INPUTS
        # --------------------------------------

        with gr.Column(elem_classes="card"):

            gr.Markdown("## 🏡 Property Details")

            area = gr.Number(
                label="Area (sq ft)",
                value=2500,
                minimum=100
            )

            bedrooms = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                value=4,
                label="Bedrooms"
            )

            bathrooms = gr.Slider(
                minimum=1,
                maximum=6,
                step=1,
                value=3,
                label="Bathrooms"
            )

            floors = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=2,
                label="Floors"
            )

            year_built = gr.Slider(
                minimum=1950,
                maximum=2026,
                step=1,
                value=2018,
                label="Year Built"
            )

        # --------------------------------------
        # RIGHT SIDE - CATEGORICAL INPUTS
        # --------------------------------------

        with gr.Column(elem_classes="card"):

            gr.Markdown("## 📍 Location & Features")

            location = gr.Dropdown(
                choices=[
                    "Downtown",
                    "Rural",
                    "Suburban",
                    "Urban"
                ],
                value="Suburban",
                label="Location"
            )

            condition = gr.Dropdown(
                choices=[
                    "Excellent",
                    "Fair",
                    "Good"
                ],
                value="Good",
                label="Condition"
            )

            garage = gr.Dropdown(
                choices=[
                    "No",
                    "Yes"
                ],
                value="Yes",
                label="Garage"
            )

            predict_btn = gr.Button(
                "💰 Predict House Price",
                variant="primary",
                size="lg"
            )

            clear_btn = gr.ClearButton(
                components=[
                    area,
                    bedrooms,
                    bathrooms,
                    floors,
                    year_built,
                    location,
                    condition,
                    garage
                ],
                value="Clear"
            )

    # ------------------------------------------
    # RESULT
    # ------------------------------------------

    gr.Markdown("## 📊 Prediction Result")

    prediction = gr.Textbox(
        label="Estimated House Price",
        placeholder="Your predicted price will appear here...",
        elem_classes="result-box",
        interactive=False
    )

    predict_btn.click(
        fn=predict_house_price,
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
        outputs=prediction
    )

    # ------------------------------------------
    # EXAMPLE
    # ------------------------------------------

    gr.Markdown("""
    ### 💡 Example

    **Area:** 2500 sq ft  
    **Bedrooms:** 4  
    **Bathrooms:** 3  
    **Floors:** 2  
    **Year Built:** 2018  
    **Location:** Suburban  
    **Condition:** Good  
    **Garage:** Yes  

    The notebook used these values as a sample and obtained a predicted price of approximately **$783,862.80**. :contentReference[oaicite:2]{index=2}
    """)

    gr.HTML("""
    <div class="footer">
        Developed by <b>Vansh</b> | Roll No. 241047 | PIET, Samalkha
    </div>
    """)


# ==========================================
# LAUNCH
# ==========================================

if __name__ == "__main__":
    app.launch()

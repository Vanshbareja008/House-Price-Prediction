import gradio as gr
import joblib
import pandas as pd


# ==========================================
# Load Trained Model
# ==========================================

model = joblib.load("house_price_prediction_model.pkl")


# ==========================================
# Encoding Maps
# ==========================================

location_map = {
    "Downtown": 0,
    "Rural": 1,
    "Suburban": 2,
    "Urban": 3
}

condition_map = {
    "Excellent": 0,
    "Fair": 1,
    "Good": 2,
    "Poor": 3
}

garage_map = {
    "No": 0,
    "Yes": 1
}


# ==========================================
# Prediction Function
# ==========================================

def predict_price(area, bedrooms, bathrooms, floors,
                  year_built, location, condition, garage):

    try:

        # Convert categorical values to encoded values
        location_encoded = location_map[location]
        condition_encoded = condition_map[condition]
        garage_encoded = garage_map[garage]

        # Create input DataFrame
        input_data = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Bathrooms": [bathrooms],
            "Floors": [floors],
            "YearBuilt": [year_built],
            "Location": [location_encoded],
            "Condition": [condition_encoded],
            "Garage": [garage_encoded]
        })

        # Predict
        prediction = model.predict(input_data)[0]

        return f"${prediction:,.2f}"

    except Exception as e:
        return f"Prediction Error: {str(e)}"


# ==========================================
# Custom CSS
# ==========================================

css = """
body {
    background: #f4f5f7;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto;
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
    margin-bottom: 5px;
    font-size: 32px;
}

.header p {
    margin: 5px;
    color: #d1d5db;
}

.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
}

.footer {
    text-align: center;
    margin-top: 20px;
    color: #6b7280;
}
"""


# ==========================================
# Gradio Interface
# ==========================================

with gr.Blocks(
    title="House Price Prediction",
    css=css
) as app:

    gr.HTML("""
    <div class="header">
        <h1>🏠 House Price Prediction</h1>
        <p>Machine Learning Based Property Price Prediction</p>
        <p>Powered by XGBoost Regression</p>
    </div>
    """)

    gr.Markdown(
        "### Enter Property Details\n"
        "Provide the house information below to estimate its market price."
    )

    with gr.Row():

        with gr.Column():

            area = gr.Number(
                label="Area (sq ft)",
                value=2500,
                minimum=600,
                maximum=5000
            )

            bedrooms = gr.Slider(
                minimum=1,
                maximum=6,
                step=1,
                value=4,
                label="Bedrooms"
            )

            bathrooms = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=3,
                label="Bathrooms"
            )

            floors = gr.Slider(
                minimum=1,
                maximum=4,
                step=1,
                value=2,
                label="Floors"
            )

        with gr.Column():

            year_built = gr.Slider(
                minimum=1970,
                maximum=2024,
                step=1,
                value=2018,
                label="Year Built"
            )

            location = gr.Dropdown(
                choices=[
                    "Downtown",
                    "Rural",
                    "Suburban",
                    "Urban"
                ],
                value="Downtown",
                label="Location"
            )

            condition = gr.Dropdown(
                choices=[
                    "Excellent",
                    "Fair",
                    "Good",
                    "Poor"
                ],
                value="Good",
                label="Property Condition"
            )

            garage = gr.Dropdown(
                choices=[
                    "Yes",
                    "No"
                ],
                value="Yes",
                label="Garage"
            )

    predict_button = gr.Button(
        "Predict House Price",
        variant="primary"
    )

    result = gr.Textbox(
        label="Predicted House Price",
        placeholder="Your predicted price will appear here...",
        elem_classes="result-box"
    )

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

    gr.Markdown("""
    <div class="footer">
        Developed by Vansh | Roll No. 241047 | PIET, Samalkha
    </div>
    """)


# ==========================================
# Launch Application
# ==========================================

if __name__ == "__main__":
    app.launch()

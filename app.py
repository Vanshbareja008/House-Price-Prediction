import os
import gradio as gr
import joblib
import pandas as pd

# ==========================================
# Load trained model
# ==========================================

model = joblib.load("house_price_prediction_model.pkl")


# ==========================================
# Prediction Function
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
# Gradio Interface
# ==========================================

with gr.Blocks(
    title="House Price Predictor"
) as demo:

    gr.Markdown(
        """
        # 🏠 House Price Predictor

        ### Predict the estimated price of a house using Machine Learning

        Enter the property details below and click **Predict House Price**.
        """
    )

    with gr.Row():

        # ----------------------------------
        # Left Column
        # ----------------------------------

        with gr.Column():

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

        # ----------------------------------
        # Right Column
        # ----------------------------------

        with gr.Column():

            year_built = gr.Number(
                label="Year Built",
                value=2018,
                minimum=1800,
                maximum=2026,
                precision=0
            )

            # Location:
            # Downtown = 0
            # Suburban  = 1
            # Urban     = 2

            location = gr.Dropdown(
                choices=[
                    ("Downtown", 0),
                    ("Suburban", 1),
                    ("Urban", 2)
                ],
                value=1,
                label="Location"
            )

            # Condition:
            # Excellent = 0
            # Fair      = 1
            # Good      = 2

            condition = gr.Dropdown(
                choices=[
                    ("Excellent", 0),
                    ("Fair", 1),
                    ("Good", 2)
                ],
                value=2,
                label="Condition"
            )

            # Garage:
            # No = 0
            # Yes = 1

            garage = gr.Dropdown(
                choices=[
                    ("No", 0),
                    ("Yes", 1)
                ],
                value=1,
                label="Garage"
            )

    # ======================================
    # Predict Button
    # ======================================

    predict_button = gr.Button(
        "🔮 Predict House Price",
        variant="primary"
    )

    # ======================================
    # Result
    # ======================================

    result = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    # ======================================
    # Button Action
    # ======================================

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

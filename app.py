import os
import gradio as gr
import joblib
import pandas as pd
import pandas_bokeh
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# Load trained model
# ==========================================

model = joblib.load("house_price_prediction_model.pkl")

# Location & Condition mapping for UI display logic
LOCATION_MAP = {0: "Downtown", 1: "Suburban", 2: "Urban"}
CONDITION_MAP = {0: "Excellent", 1: "Fair", 2: "Good"}


# ==========================================
# Core Logic & Dynamic Analytics
# ==========================================

def predict_and_analyze(
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
        # 1. Original DataFrame Creation & Model Prediction
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

        predicted_price = model.predict(input_data)[0]
        formatted_price = f"₹ {predicted_price:,.2f}"

        # 2. Dynamic Investment & Valuation Suggestions
        suggestions = []
        
        # Price per sqft evaluation
        price_per_sqft = predicted_price / area if area > 0 else 0
        suggestions.append(f"• **Unit Rate:** Estimated at **₹{price_per_sqft:,.2f}/sq ft**.")

        # Age impact
        age = 2026 - year_built
        if age > 30:
            suggestions.append("• **Age Factor:** Property is over 30 years old. Budgeting for structural updates or HVAC upgrades could increase resale value.")
        elif age < 5:
            suggestions.append("• **New Construction:** Premium pricing reflects recent build year with lower immediate maintenance overhead.")

        # Feature density check
        if area > 0 and (bedrooms + bathrooms) / area > 0.005:
            suggestions.append("• **Layout Density:** High room count relative to area. Ensure room sizes meet target buyer expectations.")
        
        if condition == 1:  # Fair condition
            suggestions.append("• **Renovation Potential:** Upgrading property condition from 'Fair' to 'Excellent' typically yields high ROI.")

        if garage == 0:
            suggestions.append("• **Feature Gap:** Adding covered parking/garage space can significantly improve buyer liquidity in suburban areas.")

        suggestion_text = "\n".join(suggestions)

        # 3. Interactive Chart 1: Estimated Feature Contribution
        # Sample relative importance metrics for visual engagement
        features = ['Area', 'Location', 'Year Built', 'Bedrooms', 'Bathrooms', 'Condition', 'Garage', 'Floors']
        importance = [35, 20, 15, 10, 8, 6, 4, 2]
        
        fig_importance = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Estimated Feature Impact on Price",
            labels={'x': 'Relative Weight (%)', 'y': 'Feature'},
            color=importance,
            color_continuous_scale="Blues"
        )
        fig_importance.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            coloraxis_showscale=False
        )

        # 4. Interactive Chart 2: Market Trend Projection
        years = [year_built, year_built + 5, year_built + 10, 2026, 2028, 2030]
        years = sorted(list(set([y for y in years if y <= 2030])))
        
        # Simulated appreciation curve around estimated price
        base_val = predicted_price * 0.7
        trend_prices = [base_val * ((1.05) ** (i)) for i in range(len(years))]
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=years, 
            y=trend_prices, 
            mode='lines+markers',
            line=dict(color='#2b6cb0', width=3),
            marker=dict(size=8, color='#1a202c'),
            name="Valuation Trend"
        ))
        fig_trend.update_layout(
            title="Property Appreciation Trend (Estimated)",
            xaxis_title="Year",
            yaxis_title="Valuation (₹)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Plus Jakarta Sans, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return formatted_price, suggestion_text, fig_importance, fig_trend

    except Exception as e:
        return f"Error: {str(e)}", "", None, None


# ==========================================
# Custom CSS Styling
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
    font-size: 2.6rem;
    font-weight: 700;
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
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}
.result-box textarea {
    font-size: 2rem !important;
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
# Gradio Interface Build
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
            <p>Predict real estate market pricing and evaluate property metrics through ML analytics</p>
        </div>
        """
    )

    with gr.Column(elem_classes=["card-container"]):
        with gr.Row():

            # Left Input Column
            with gr.Column():
                gr.Markdown("### 📐 Property Structure")
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

            # Right Input Column
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

        # Trigger Action
        with gr.Row():
            predict_button = gr.Button(
                "✨ Calculate Valuation & Insights",
                variant="primary",
                elem_classes=["predict-btn"]
            )

    # Output Display Section
    with gr.Column(elem_classes=["card-container"]):
        gr.Markdown("### 📊 Valuation & Investment Insights")
        
        with gr.Row():
            with gr.Column(scale=1):
                result = gr.Textbox(
                    label="Estimated Valuation",
                    interactive=False,
                    elem_classes=["result-box"]
                )
                suggestions_box = gr.Markdown(
                    label="Smart Suggestions",
                    value="*Click 'Calculate Valuation' to generate customized suggestions.*"
                )

        # Interactive Graphical Analytics
        with gr.Row():
            chart_importance = gr.Plot(label="Feature Weight Breakdown")
            chart_trend = gr.Plot(label="Appreciation Trend")

    # Dynamic Interaction
    predict_button.click(
        fn=predict_and_analyze,
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
        outputs=[
            result,
            suggestions_box,
            chart_importance,
            chart_trend
        ]
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

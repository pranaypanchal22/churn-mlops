# Frontend (Streamlit)

A beginner-friendly form that sends one customer profile to the API and shows the estimate it
returns. **The frontend contains no model and no copy of the model logic.** It builds its form from
the API's `/model-info` input schema, and every prediction comes from `POST /predict`.

```bash
# terminal 1: API
.venv/Scripts/python -m uvicorn churn.api.app:app --port 8000
# terminal 2: UI, then open http://localhost:8501
.venv/Scripts/python -m streamlit run frontend/app.py
```

| Variable | Default | Purpose |
|---|---|---|
| `CHURN_API_URL` | `http://localhost:8000` | API base URL |
| `CHURN_API_TIMEOUT` | `5` | Seconds before a request counts as failed |

The frontend's own dependencies are listed in [`frontend/requirements.txt`](../frontend/requirements.txt)
(Streamlit and httpx only). Its Docker image does not need scikit-learn or pandas.

## User flow

1. On load, the app fetches `/model-info`, which is cached for 60 s. The form opens with a made-up sample
   profile already filled in, so the demo works straight away. Two sample buttons switch between profiles.
2. Inputs are grouped as Account, Household, Phone, and Internet. Numbers use number inputs
   bounded by the schema's min/max. Options with three or fewer choices use radio buttons. Payment
   method uses a dropdown. Every option comes from the schema served by the API.
3. **Estimate churn risk** sends the form to `POST /predict` while a spinner shows.
4. The result card shows the probability, the High/Low risk level, the threshold, the model version, and
   the API's "estimate, not a guarantee" note.

## Validation and errors

| Situation | What the user sees |
|---|---|
| Field left empty | "Please fill in: …" and no request is sent |
| Combination never seen in the data (e.g. internet service "No" with an add-on "Yes", or total < monthly) | A non-blocking hint. The request is still sent, because the API decides validity. |
| API rejects input (422) | Each rejected field, with its friendly label and the API's message |
| API down or timed out | "The prediction service isn't reachable", how to start it, and a **Retry** button |
| API up but no model promoted (503) | "No model loaded", pointing to train/promote, with Retry |
| Other server error | A generic retry message |

## Privacy

There are no fields for names, IDs, or contact details. Payloads contain only the 18 schema features. The app
logs nothing about inputs or results, and the API rejects unknown fields.

## Layout

The page is centered. On desktop each section shows two columns, and on narrow screens Streamlit stacks
them into one. Checked at desktop width and at 375 px, with no horizontal scrolling.

## Screenshots (for the README)

1. Start the API and UI as above. Use a desktop browser window about 1280 px wide.
2. Capture: (a) the form with the default sample, (b) the result card after **Estimate churn
   risk**, (c) the "service isn't reachable" state after stopping the API.
3. Save them as `docs/images/frontend_1_form.jpg`, `frontend_2_result.jpg`, and `frontend_3_api_down.jpg`.
   Use a new filename when replacing an image: GitHub and browsers cache README images by URL.

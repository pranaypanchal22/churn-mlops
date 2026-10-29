# Frontend: churn risk explorer (Streamlit)

A two-step page: enter a customer's details, press **Get prediction**, and see what the model says
about them, and *why*. **The frontend
contains no model and no copy of the model logic.** It builds its inputs from the API's
`/model-info` input schema, and every number it shows comes from `POST /predict` or
`POST /predict/batch`.

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

Dependencies are in [`frontend/requirements.txt`](../frontend/requirements.txt): Streamlit and
httpx. Charts use Altair, which ships with Streamlit. The image doesn't need scikit-learn.

## Layout and flow

Two columns on desktop: **1 · Customer details** on the left, **2 · Prediction** on the right.

- **Customer details.**
  - A sample picker with three made-up customers. Picking one fills the form and clears any old result.
  - A form with Account (open), Household, Phone, and Internet sections. Closed sections already
    hold values.
  - Tenure is a slider and charges are number inputs. Options with three or fewer choices use radio buttons,
    and payment method uses a dropdown. Every option and limit comes from the API's schema, and short help text explains each field.
  - A full-width **🔮 Get prediction** button at the bottom.
- **Nothing is predicted until you submit.** It's a Streamlit form, so edits don't apply until the button
  is pressed. The results therefore always match the details that were sent, and a caption says so.
- **Prediction** (after submit):
  - **Model's score**, **Risk level**, and **Customers this line flags** (the validation-set share at or
    above the threshold, from `/model-info → reference`);
  - a **gauge** with the score and the dashed decision line;
  - a **plain-language verdict**: "well above / just above / just below / well below the line";
  - notes on anything the form adjusted on submit;
  - the API's "estimate, not a guarantee" note and the model version.

  Before the first submit, this panel explains what you'll get.
- **Tabs:**
  - **🔀 What could change it?** One bar per single change to a *changeable* lever: contract,
    payment method, paperless billing, online security, tech support. Bars are coloured by which side of the line
    they land on, sorted with the biggest drop first, and the largest drop is spelled out. Personal
    attributes such as age or household are deliberately *not* offered as levers.
  - **📈 Over time.** The same customer and plan at tenures 0–72 months (the range in the training data), with total
    charges recalculated. It says where the score crosses the line.
  - **🧠 About the model.** Test-set recall and precision, the training churn rate, the threshold rule, the
    top model-level signals (labelled as not causal), and known limits.
  - **❓ How it works.** A five-step explanation for non-technical viewers.

## Keeping inputs realistic (applied on submit)

- **"No" for internet (or phone)** sets the matching add-ons to "No internet service" (or "No phone
  service") and locks them. Choosing a service again resets them to "No". This mirrors the training
  data, where those values always go together. The result says what was changed.
- **"Work it out from tenure × monthly charges"** calculates total charges when you submit, and the result shows
  the value used.
- The remaining consistency check (total lower than one month's charges) appears as a non-blocking hint.

## What-if requests and monitoring

The what-if and over-time charts score about 35 hypothetical variants of the customer on each submit.
They're sent in one `/predict/batch` call with the header `X-Churn-Purpose: exploration`. The API
scores them normally but **keeps them out of drift monitoring and the prediction metrics**, and counts them
separately (`churn_exploration_rows_total`). Without this, every submit would look like a sudden shift in real
customers. Tests check that loading the page adds nothing to drift monitoring, and that one submit adds
exactly one row, including after the page redraws.

## Errors

| Situation | What the user sees |
|---|---|
| Field missing | "Please fill in: …" and no request is sent |
| API rejects input (422) | Each rejected field with its friendly label and the API's message |
| API down on page load | "The prediction service isn't reachable", how to start it, and a **Retry** button |
| API down when submitting | The same message, telling you to press **Get prediction** again |
| API up but no model promoted (503) | "No model loaded", pointing to train/promote, with Retry |
| Scenario scoring fails | The main result still shows. The tab shows a warning. |

## Privacy

There are no fields for names, IDs, or contact details. Payloads contain only the 18 schema features. The app
logs and stores nothing about inputs or results, and the API rejects unknown fields.

## Screen sizes

Two columns on desktop. On phones (checked at 375 px) the form comes first with a full-width button, and
the prediction stacks below it. A "Prediction ready" notification appears after submitting. No horizontal scrolling.

## Screenshots (for the README)

Captured from the Docker stack in a 1200×750 viewport: `docs/images/explorer_v3_1_form.jpg`,
`explorer_v3_2_result.jpg`, `explorer_v3_3_what_if.jpg`. **Use a new filename when replacing an image**,
because GitHub and browsers cache README images by URL.

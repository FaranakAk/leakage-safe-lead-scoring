"""Lead Priority Scorer: Streamlit front end for the frozen lead-ranking model.

Run:  streamlit run app.py

Loads only the committed deployment artifact (artifacts/models/lead-scorer-1.0.0-deploy.joblib).
No training, no dataset download, no fitting at runtime.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import app_logic as app  # noqa: E402

st.set_page_config(page_title="Lead Priority Scorer", page_icon=":material/leaderboard:", layout="wide")

st.markdown(
    """
    <style>
      .priority-card {border: 1px solid rgba(128,128,128,0.25); border-radius: 12px; padding: 14px 18px;}
      .priority-label {font-size: 0.85rem; opacity: 0.75; margin-bottom: 4px;}
      .priority-value {display: inline-block; font-size: 1.9rem; font-weight: 700; padding: 2px 14px;
                       border-radius: 8px; line-height: 1.3;}
      .priority-High {background: #184f95; color: #ffffff;}
      .priority-Medium {background: #3987e5; color: #ffffff;}
      .priority-Low {background: #cde2fb; color: #0d366b;}
      .priority-sub {font-size: 0.85rem; opacity: 0.75; margin-top: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

BAND_MEANING = {
    "High": "Top 20% of reference leads",
    "Medium": "Next 30% of reference leads",
    "Low": "Bottom 50% of reference leads",
}
HIGHER, LOWER = "#2a78d6", "#e34948"


@st.cache_resource
def get_scorer():
    return app.load_scorer()


def ordinal(n: float) -> str:
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def priority_card(title: str, band: str, subtitle: str) -> None:
    st.markdown(
        f"""<div class="priority-card"><div class="priority-label">{title}</div>
        <span class="priority-value priority-{band}">{band}</span>
        <div class="priority-sub">{subtitle}</div></div>""",
        unsafe_allow_html=True,
    )


def explanation_chart(explanation: pd.DataFrame):
    data = explanation.reindex(explanation["contribution"].abs().sort_values(ascending=False).index).head(8).copy()
    data["factor"] = data["field"] + ": " + data["value"]
    data["direction"] = data["contribution"].map(lambda c: "Pushed score higher" if c > 0 else "Pushed score lower")
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=16)
        .encode(
            x=alt.X("contribution:Q", title="Contribution to the model's ranking score"),
            y=alt.Y(
                "factor:N",
                sort=alt.EncodingSortField("contribution", order="descending"),
                title=None,
                axis=alt.Axis(labelLimit=320),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Pushed score higher", "Pushed score lower"], range=[HIGHER, LOWER]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("contribution:Q", title="Contribution", format="+.2f")],
        )
        .properties(height=alt.Step(26))
    )


scorer = get_scorer()

st.title("Lead Priority Scorer")
st.markdown(
    "Ranks bank term-deposit leads **before the call is dialled**, so a sales team can call the most promising "
    "leads first."
)

with st.expander("About this model"):
    st.markdown(
        """
- **What it does:** ranks leads for outreach prioritisation using only information known before the call
  (client profile, credit status, planned contact channel and weekday, contact history). Call duration and the
  call's outcome are never used.
- **How it was tested:** forward in time. It was trained on May 2008 – March 2009 campaigns and checked once on
  unseen later campaigns (May 2009 – November 2010).
- **What held up:** the ranking stayed useful after the market changed. In the later period the top 20% of
  ranked leads converted at about 1.8 times the average rate.
- **What did not:** probability calibration deteriorated. The overall conversion rate more than doubled, and
  the model's probabilities stayed far too low.
- **Before real use:** monitor results and recalibrate or retrain on recent outcome data.
- **Data:** UCI Bank Marketing dataset (Moro, Cortez & Rita, 2014), CC BY 4.0. A prioritisation aid, not a guarantee
  of conversion.
        """
    )

single_tab, batch_tab = st.tabs(["Single lead", "Batch scoring"])

# --- single lead -----------------------------------------------------------------------------------

with single_tab:
    inputs_col, results_col = st.columns([5, 6], gap="large")

    with inputs_col:

        def pick(field: str, default: str, key: str | None = None):
            options = app.categories(scorer, field)
            raw_column = app.CATEGORICAL_FIELDS[field]
            return st.selectbox(
                app.FIELD_LABELS[field],
                options,
                index=options.index(default) if default in options else 0,
                format_func=lambda code: app.display_label(raw_column, code),
                key=key or field,
            )

        with st.container(border=True):
            st.markdown("**Client profile**")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", min_value=17, max_value=100, value=35, step=1, key="age")
            with c2:
                job = pick("job", "management")
            with c1:
                marital = pick("marital_status", "married")
            with c2:
                education = pick("education", "university.degree")

        with st.container(border=True):
            st.markdown("**Credit and loans**")
            c1, c2, c3 = st.columns(3)
            with c1:
                default = pick("credit_default", "no")
            with c2:
                housing = pick("housing_loan", "yes")
            with c3:
                loan = pick("personal_loan", "no")

        with st.container(border=True):
            st.markdown("**Contact plan and history**")
            c1, c2 = st.columns(2)
            with c1:
                contact = pick("contact_channel", "cellular")
            with c2:
                weekday = pick("call_weekday", "thu")
            c1, c2 = st.columns(2)
            earlier_calls = c1.number_input(
                app.FIELD_LABELS["earlier_calls_this_campaign"], min_value=0, max_value=60, value=0, step=1,
                key="earlier_calls_this_campaign", help="Calls already made to this client in the current campaign.",
            )
            earlier_contacts = c2.number_input(
                app.FIELD_LABELS["earlier_campaign_contacts"], min_value=0, max_value=10, value=0, step=1,
                key="earlier_campaign_contacts", help="Contacts with this client during previous campaigns.",
            )
            outcomes = app.outcome_options(scorer, int(earlier_contacts))
            outcome = st.selectbox(
                app.FIELD_LABELS["previous_campaign_outcome"],
                outcomes,
                format_func=lambda code: app.display_label("poutcome", code),
                disabled=len(outcomes) == 1,
                key=f"previous_campaign_outcome_{'none' if earlier_contacts == 0 else 'some'}",
                help="Only available when there were contacts in earlier campaigns.",
            )

    lead = {
        "age": int(age), "job": job, "marital_status": marital, "education": education, "credit_default": default,
        "housing_loan": housing, "personal_loan": loan, "contact_channel": contact, "call_weekday": weekday,
        "earlier_calls_this_campaign": int(earlier_calls), "earlier_campaign_contacts": int(earlier_contacts),
        "previous_campaign_outcome": outcome,
    }

    with results_col:
        problems = app.validate_lead(scorer, lead)
        if problems:
            st.error("This lead cannot be scored: " + "; ".join(problems))
        else:
            result = app.score_lead(scorer, lead)
            band = result["historical_reference_priority"]
            c1, c2 = st.columns(2)
            with c1:
                priority_card("Historical reference priority", band, BAND_MEANING[band])
            with c2:
                st.metric(
                    "Historical percentile",
                    ordinal(result["historical_percentile"]),
                    help="Share of reference leads (April–May 2009) that this lead scores at or above.",
                    border=True,
                )
            st.caption(
                f"Ranks above about {result['historical_percentile']:.0f}% of leads from the reference period."
            )
            if result["outside_training_range"]:
                st.caption(
                    ":material/info: A numeric input is outside the range seen in training; the score extrapolates."
                )

            st.markdown("##### Why this lead ranks where it does")
            higher, lower = app.top_factors(result["explanation"])
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Pushed the model score higher**")
                st.markdown("\n".join(f"- {item}" for item in higher) or "- None of note")
            with c2:
                st.markdown("**Pushed the model score lower**")
                st.markdown("\n".join(f"- {item}" for item in lower) or "- None of note")
            st.altair_chart(explanation_chart(result["explanation"]), width="stretch")
            st.caption(
                "Contributions show what the model associates with a higher or lower ranking score, compared with an "
                "average lead from its training period. They are associations, not causes."
            )

            with st.container(border=True):
                st.markdown(
                    f"<span style='opacity:0.8'>Historical calibrated estimate: "
                    f"<b>{result['historical_calibrated_estimate']:.0%}</b></span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    ":material/warning: Secondary figure. This calibration was fitted on April–May 2009 and drifted "
                    "badly in later data, where actual conversion was about twice the estimate. Do not read it as a "
                    "current conversion probability without recent outcome data."
                )

            with st.expander("Technical details"):
                st.write(
                    {
                        "ranking_score (log-odds)": round(result["ranking_score"], 4),
                        "high_priority_cutoff": round(scorer.cutoffs["high_min"], 4),
                        "medium_priority_cutoff": round(scorer.cutoffs["medium_min"], 4),
                        "explanation_baseline": round(scorer.explanation_base_value, 4),
                        "reference_period": "April–May 2009 (4,977 leads)",
                        "model": scorer.metadata["model"],
                        "artifact_version": scorer.metadata["version"],
                    }
                )

# --- batch scoring ----------------------------------------------------------------------------------

with batch_tab:
    st.markdown(
        "Upload a CSV of leads. **Batch priority** ranks leads within your file: the top 20% are High, the next "
        "30% Medium, the remaining 50% Low. **Historical reference priority** compares each lead with the "
        "April–May 2009 reference period."
    )
    c1, c2, _ = st.columns([1, 1, 2])
    c1.download_button(
        "Download template", app.template_frame().to_csv(index=False), "lead_template.csv", "text/csv",
        icon=":material/download:",
    )
    c2.download_button(
        "Download example leads", app.EXAMPLE_CSV.read_bytes(), "example_leads.csv", "text/csv",
        icon=":material/download:",
    )

    uploaded = st.file_uploader("Leads CSV", type=["csv"], key="upload")
    use_example = st.toggle("Use the example file (200 leads) instead", key="use_example")

    source = None
    if uploaded is not None:
        source = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=None, engine="python")
    elif use_example:
        source = pd.read_csv(app.EXAMPLE_CSV)

    if source is not None:
        try:
            leads = app.normalise_upload(source)
        except ValueError as error:
            st.error(str(error))
            leads = None
        if leads is not None:
            valid, invalid = app.split_valid(scorer, leads)
            scored = app.score_batch_table(scorer, valid)
            counts = scored["batch_priority"].value_counts()
            m = st.columns(4)
            m[0].metric("Leads scored", f"{len(scored):,}", border=True)
            for column, band in zip(m[1:], ("High", "Medium", "Low")):
                column.metric(f"Batch priority: {band}", f"{counts.get(band, 0):,}", border=True)
            if len(invalid):
                with st.expander(f"{len(invalid)} row(s) could not be scored"):
                    st.dataframe(invalid, width="stretch")
            if len(scored) and len(scored) < 10:
                st.caption("With fewer than 10 leads, batch bands are coarse; tied scores share the higher band.")

            st.dataframe(
                scored,
                width="stretch",
                hide_index=True,
                column_config={
                    "batch_percentile": st.column_config.NumberColumn(format="%.1f"),
                    "historical_percentile": st.column_config.NumberColumn(format="%.1f"),
                    "historical_calibrated_estimate": st.column_config.NumberColumn(
                        "historical_calibrated_estimate (secondary)", format="%.3f"
                    ),
                },
            )
            st.download_button(
                "Download scored CSV", scored.to_csv(index=False), "scored_leads.csv", "text/csv",
                icon=":material/download:", type="primary",
            )
            st.caption(
                "The historical calibrated estimate is secondary: calibration drifted in later data. Rank and "
                "prioritise by batch priority. Extra columns in your file (including any outcome or call-duration "
                "fields) are ignored and not included in the download."
            )

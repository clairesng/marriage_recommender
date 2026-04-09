import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Streamlit page config must be the first Streamlit command.
st.set_page_config(page_title="Marriage Recommender", page_icon="💍")

# 1. Load the model
@st.cache_resource
def load_marriage_model():
    model = CatBoostClassifier()
    model_path = Path(__file__).resolve().parent / "catboost_run4_best_model.cbm"
    model.load_model(str(model_path))
    return model

model = load_marriage_model()

RUN4_FEATURES = [
    "age_gap",
    "Our Family Backgrounds:",
    "meet_clean",
    "How long was your relationship before marriage?",
    "Did you attend Premarital Counselling?",
    'Do you believe in "Soulmates" or "The One"?',
    "What were your views on children?",
    "On political views:",
    "inlaw_min_band",
    "Our household is primarily:",
    "On managing household finances:",
    "Do you share a key common interest?",
    "Do you approve of each other's social circle?",
    "values_overlap_count",
]

# Keep finance labels centralized so UI text and scoring keys always match exactly.
FINANCE_WINDFALL_OPTIONS = [
    "I would save almost all of the money",
    "I would split it: save some and use some for a holiday or treat",
    "I would spend most of it on experiences or big purchases",
]

FINANCE_PERSONA_OPTIONS = [
    "Frugal bargain hunter — I like to save and hunt for the best deals",
    "I like to indulge because money can be earned back",
    "I tend to let my partner make the financial decisions",
]

FINANCE_OVERSPEND_OPTIONS = [
    "Cut back next month and rebalance the budget",
    "Review and make small adjustments",
    "Keep living as usual and figure it out later",
]

FINANCE_WINDFALL_MAP = {
    FINANCE_WINDFALL_OPTIONS[0]: 1,
    FINANCE_WINDFALL_OPTIONS[1]: 2,
    FINANCE_WINDFALL_OPTIONS[2]: 3,
}

FINANCE_PERSONA_MAP = {
    FINANCE_PERSONA_OPTIONS[0]: 1,
    FINANCE_PERSONA_OPTIONS[1]: 3,
    FINANCE_PERSONA_OPTIONS[2]: 2,
}

FINANCE_OVERSPEND_MAP = {
    FINANCE_OVERSPEND_OPTIONS[0]: 1,
    FINANCE_OVERSPEND_OPTIONS[1]: 2,
    FINANCE_OVERSPEND_OPTIONS[2]: 3,
}


@st.cache_resource
def load_run4_preprocessor():
    feature_store_path = Path(__file__).resolve().parents[1] / "divorced_feature_store.csv"
    df = pd.read_csv(feature_store_path)
    X = df[RUN4_FEATURES].copy()

    numeric_features = [c for c in RUN4_FEATURES if pd.api.types.is_numeric_dtype(X[c])]
    categorical_features = [c for c in RUN4_FEATURES if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    preprocessor.fit(X)
    return preprocessor


pre4 = load_run4_preprocessor()


@st.cache_data
def load_eda_data():
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "divorced.csv"
    feature_store_path = root / "divorced_feature_store.csv"
    raw_df = pd.read_csv(raw_path, encoding="latin1")
    feature_df = pd.read_csv(feature_store_path)
    return raw_df, feature_df


def render_project_hub():
    st.divider()
    st.header("📘 About this project")

    tab_why, tab_eda = st.tabs(["Why this project", "EDA Dashboard"])

    with tab_why:
        st.subheader("What this project is about")
        st.write(
            "This 'Marriage Recommender' is designed to help couples reflect on their relationship and work on areas that matter in the long-term." \
        )
        st.write(
            "It is based on anonymous survey data collected from married and divorced couples in Singapore. "
        )
    with tab_why:
        st.subheader("Why I built this")
        st.write(
            "I wanted to create a tool that uses data and machine learning to provide insights that can help couples build stronger relationships and make thoughtful decisions about their future together."
        )
        st.write("Plus, I thought it would be fun to apply machine learning to the topic of love and relationships, which is something that affects us all but is often seen as more of an art than a science. Not to mention the recent news about Singapore's falling birth rate and marriages."
        )

        st.subheader("About the ML model")
        st.write(
            "After testing several machine-learning approaches, the CatBoost model was chosen because it performed best on this dataset with the highest accuracy and lowest error rate. "
            "CatBoost is good at spotting useful patterns in questionnaire-style data, especially when answers are mostly categorical."
        )
        st.write(
            "The 14 factors used were selected due to their strength in the model in being able to predict marriage outcomes."
        )

        st.subheader("The 14 relationship factors used in the model")
        st.markdown(
            "1. Age difference between partners\n"
            "2. Family background (whether parents are married or divorced)\n"
            "3. How the couple met\n"
            "4. Length of relationship before marriage\n"
            "5. Premarital counselling\n"
            "6. Belief in soulmates\n"
            "7. Views on having children\n"
            "8. Political views\n"
            "9. Quality of relationship with in-laws\n"
            "10. Main household income setup\n"
            "11. Financial alignment\n"
            "12. Key common interest\n"
            "13. Whether they approve of each other's social circles\n"
            "14. Personal values overlap"
        )

        st.subheader("What this app is not")
        st.info(
            "This app is not designed to be deterministic, and does not replace counselling or professional advice."
        )

    with tab_eda:
        st.subheader("Exploratory Data Analysis")

        raw_df, feature_df = load_eda_data()
        status_col = "What is your current marital status?"
        status_display_map = {
            "Married or Widowed": "Married",
            "Divorced or Annulled": "Divorced",
        }

        status_series = None
        if status_col in feature_df.columns:
            status_series = feature_df[status_col]
        elif status_col in raw_df.columns and len(raw_df) == len(feature_df):
            status_series = raw_df[status_col]

        sample_n = len(feature_df)
        feature_n = len(RUN4_FEATURES)

        c1, c2 = st.columns(2)
        c1.metric("Samples", f"{sample_n:,}")
        c2.metric("Features used", f"{feature_n}")

        if status_series is not None:
            st.markdown("### Marital-status breakdown")
            status_plot = (
                status_series.map(status_display_map)
                .fillna(status_series)
                .value_counts()
                .rename_axis("status")
                .to_frame("count")
            )
            st.bar_chart(status_plot)

        st.markdown("### Explore one relationship area")
        feature_labels = {
            "age_gap": "Age difference between partners",
            "Our Family Backgrounds:": "Family background",
            "meet_clean": "How the couple met",
            "How long was your relationship before marriage?": "Relationship length before marriage",
            "Did you attend Premarital Counselling?": "Premarital counselling",
            'Do you believe in "Soulmates" or "The One"?': "Belief in soulmates",
            "What were your views on children?": "Views on children",
            "On political views:": "Political views",
            "inlaw_min_band": "Relationship with in-laws",
            "Our household is primarily:": "Household income setup",
            "On managing household finances:": "Managing household finances",
            "Do you share a key common interest?": "Shared key common interest",
            "Do you approve of each other's social circle?": "Approval of each other's social circle",
            "values_overlap_count": "Overlap in top life values",
        }

        selected_feature = st.selectbox(
            "Pick a feature",
            RUN4_FEATURES,
            format_func=lambda x: feature_labels.get(x, x),
        )

        def clean_dashboard_value(value):
            if pd.isna(value):
                return "(Missing)"
            text = str(value).strip()
            text = text.replace("â\x80\x93", "-")
            text = text.replace("â€“", "-")
            text = text.replace("–", "-")
            return text if text else "(Missing)"

        if selected_feature in feature_df.columns:
            st.markdown(f"**Selected area:** {feature_labels.get(selected_feature, selected_feature)}")
            tmp = feature_df.copy()
            tmp["_status"] = status_series.values if status_series is not None else "All respondents"

            is_numeric = pd.api.types.is_numeric_dtype(tmp[selected_feature])
            viz_series = tmp[selected_feature]

            if is_numeric and viz_series.dropna().nunique() > 8:
                viz_series = pd.cut(viz_series, bins=6, duplicates="drop")
                feature_title = f"{feature_labels.get(selected_feature, selected_feature)} (binned)"
            else:
                feature_title = feature_labels.get(selected_feature, selected_feature)

            if is_numeric:
                viz_series = viz_series.astype("object")
                viz_series = viz_series.where(viz_series.notna(), "(Missing)")
                viz_series = viz_series.astype(str).str.strip().replace({"": "(Missing)"})
            else:
                viz_series = viz_series.map(clean_dashboard_value)

            if selected_feature == "How long was your relationship before marriage?":
                viz_series = viz_series.replace({
                    "6 months-1 year": "6 months - 1 year",
                    "2-4 years": "2 - 4 years",
                    "1-2 years": "1 - 2 years",
                })

            if status_series is not None:
                status_clean = tmp["_status"].map(status_display_map).fillna(tmp["_status"])
                ctab = pd.crosstab(status_clean, viz_series).reindex(["Married", "Divorced"]).fillna(0)

                st.markdown("#### Married vs Divorced counts")
                st.dataframe(ctab.astype(int))
                st.bar_chart(ctab.T.astype(int))
            else:
                overall_counts = viz_series.value_counts(dropna=False).to_frame("count")
                st.markdown("#### Overall distribution")
                st.dataframe(overall_counts)
                st.bar_chart(overall_counts)
        else:
            st.warning("This feature is not available in the feature-store file.")



# Session state
if "partner_answers" not in st.session_state:
    st.session_state.partner_answers = {"Girlfriend": None, "Boyfriend": None}
else:
    pa = st.session_state.partner_answers
    if isinstance(pa, dict) and ("Girlfriend" not in pa or "Boyfriend" not in pa):
        st.session_state.partner_answers = {
            "Girlfriend": pa.get("Girlfriend", pa.get("Partner A", None)),
            "Boyfriend": pa.get("Boyfriend", pa.get("Partner B", None)),
        }

if "show_project_hub" not in st.session_state:
    st.session_state.show_project_hub = False

if st.session_state.show_project_hub:
    if st.button("Back to the quiz"):
        st.session_state.show_project_hub = False
        st.rerun()
    render_project_hub()
    st.stop()

if st.button("Learn more about this project"):
    st.session_state.show_project_hub = True
    st.rerun()

# 2. UI Configuration (quiz page only)
st.title("💍 The Marriage Compatibility Lab")
st.markdown("### Two-player quiz")
st.caption("Answer on the same device. Responses are saved separately and kept hidden.")


def combine_meet_clean(a_raw, b_raw):
    meet_map = {
        "Online (e.g. Dating App)": "Online",
        "Mutual friends": "Social",
        "Social event/Hobby group": "Social",
        "Family introduction": "Introduction",
        "Workplace": "Work/School",
        "School": "Work/School",
        "Others": "Other",
    }
    a = meet_map.get(a_raw, "Other")
    b = meet_map.get(b_raw, "Other")
    return a if a == b else "Other"


def combine_relationship_length(a_len, b_len):
    order = ["<6 months", "6 months–1 year", "1–2 years", "2–4 years", "4+ years"]
    return order[min(order.index(a_len), order.index(b_len))]


def combine_soulmates(a_choice, b_choice):
    soulmate_yes = "What is meant to be for you will always be for you"
    if a_choice == soulmate_yes and b_choice == soulmate_yes:
        return "Yes - we both do"
    if a_choice != soulmate_yes and b_choice != soulmate_yes:
        return "No - we don't"
    return "Half - one of us does"


def combine_children(a_choice, b_choice):
    if a_choice == "I want children" and b_choice == "I want children":
        return "Both wanted children"
    if a_choice == "I don't want children" and b_choice == "I don't want children":
        return "Both didn't want children"
    if a_choice == "I'm undecided" and b_choice == "I'm undecided":
        return "Both ambivalent or undecided"
    return "Mixed - One spouse wanted children"


def combine_politics(a_choice, b_choice):
    if "friction" in a_choice.lower() or "friction" in b_choice.lower():
        return "Our different views cause conflict"
    if "avoid" in a_choice.lower() and "avoid" in b_choice.lower():
        return "We both avoid discussing politics"
    if "eye to eye" in a_choice.lower() and "eye to eye" in b_choice.lower():
        return "We share the same views"
def finance_style_score(windfall_choice, persona_choice, overspend_choice):
    if (
        windfall_choice not in FINANCE_WINDFALL_MAP
        or persona_choice not in FINANCE_PERSONA_MAP
        or overspend_choice not in FINANCE_OVERSPEND_MAP
    ):
        raise ValueError(
            "Unexpected finance option text. Please click 'Start new quiz (reset both responses)' and resubmit."
        )

    s1 = FINANCE_WINDFALL_MAP[windfall_choice]
    s2 = FINANCE_PERSONA_MAP[persona_choice]
    s3 = FINANCE_OVERSPEND_MAP[overspend_choice]
    return (s1 + s2 + s3) / 3


def combine_finances(a_score, b_score, household_aligned=True, full_money_aligned=False):
    gap = abs(a_score - b_score)
    # Softer household mismatch penalty: nudge the gap slightly instead of forcing a full-level downgrade.
    # If both partners gave exactly the same three money-related answers, do not apply penalty.
    if (not household_aligned) and (not full_money_aligned):
        gap += 0.35

    if gap <= 0.5:
        return "Mostly aligned"
    if gap <= 1.25:
        return "Half aligned, half in disagreement"
    return "Rarely aligned"


def apply_money_persona_alignment(a_score, b_score, a_persona, b_persona):
    defer_text = FINANCE_PERSONA_OPTIONS[2]
    adj_a, adj_b = a_score, b_score

    if a_persona == defer_text and b_persona != defer_text:
        adj_a = b_score
    if b_persona == defer_text and a_persona != defer_text:
        adj_b = a_score

    return adj_a, adj_b


def combine_family_background(a_parent, b_parent):
    if a_parent == "Parents are Married" and b_parent == "Parents are Married":
        return "Both our parents are Married"
    if a_parent == "Parents are Divorced" and b_parent == "Parents are Divorced":
        return "Both our parents are Divorced"
    return "One set of parents are Married, the other Divorced"


def combine_household(a_choice, b_choice):
    # Use Boyfriend's selected option when mismatched (deterministic tie-breaker)
    if a_choice == b_choice:
        return a_choice, True
    return b_choice, False


def is_full_money_alignment(gf_answers, bf_answers):
    return (
        gf_answers["finance_windfall"] == bf_answers["finance_windfall"]
        and gf_answers["finance_persona"] == bf_answers["finance_persona"]
        and gf_answers["finance_overspend"] == bf_answers["finance_overspend"]
    )


def inlaw_relationship_score(holiday_choice, support_choice):
    holiday_map = {
        "I'm happy to go and feel great about it": 5,
        "I'm neutral or indifferent": 3,
        "I'd rather not go and feel uncomfortable about it": 1,
    }
    support_map = {
        "I'd do it willingly and warmly": 5,
        "I'd do it if needed": 3,
        "I'd avoid it if possible": 1,
    }
    return (holiday_map[holiday_choice] + support_map[support_choice]) / 2


def inlaw_score_to_label(score):
    if score >= 4.5:
        return "Very Good"
    if score >= 3.5:
        return "Good"
    if score >= 2.5:
        return "Neutral"
    if score >= 1.5:
        return "Bad"
    return "Very Bad"


def common_interest_score(weekend_choice, convo_choice, activity_overlap_count):
    weekend_map = {
        "We naturally choose at least one activity we both enjoy": 2,
        "We can find overlap with some effort": 1,
        "We mostly do separate things": 0,
    }
    convo_map = {
        "We have recurring topics we both get excited about": 2,
        "Sometimes yes, sometimes no": 1,
        "We struggle to find shared topics": 0,
    }
    overlap_bonus = 2 if activity_overlap_count >= 2 else (1 if activity_overlap_count == 1 else 0)
    return weekend_map[weekend_choice] + convo_map[convo_choice] + overlap_bonus


def combine_common_interest(a_score, b_score, activity_overlap_count):
    if activity_overlap_count >= 2 and min(a_score, b_score) >= 3:
        return "Yes"
    if a_score >= 5 and b_score >= 5:
        return "Yes"
    return "No"


def social_circle_score(invite_choice, feeling_choice):
    invite_map = {
        "Happy and comfortable": 2,
        "Okay, depends on context and the friends": 1,
        "Reluctant or drained": 0,
    }
    feeling_map = {
        "I like them": 2,
        "I only like some of them": 1,
        "I don't really like them": 0,
    }
    return (invite_map[invite_choice] + feeling_map[feeling_choice]) / 2


def combine_social_circle(a_score, b_score):
    couple_avg = (a_score + b_score) / 2
    if couple_avg >= 1.5:
        return "Yes"
    if couple_avg >= 0.75:
        return "Somewhat"
    return "No"


def inlaw_band(a_rating, b_rating):
    inlaw_ranks = {"Very Bad": 1, "Bad": 2, "Neutral": 3, "Good": 4, "Very Good": 5}
    min_score = min(inlaw_ranks[a_rating], inlaw_ranks[b_rating])
    if min_score <= 2:
        return "Low"
    if min_score == 3:
        return "Medium"
    return "High"


def get_stable_class_index(classes):
    class_text = [str(c).strip().lower() for c in classes]

    for i, txt in enumerate(class_text):
        if any(k in txt for k in ["married", "widowed", "stable"]):
            return i

    for i, txt in enumerate(class_text):
        if any(k in txt for k in ["divorce", "divorced", "annulled", "annul"]):
            return 1 - i if len(classes) == 2 else 0

    if 0 in list(classes):
        return list(classes).index(0)
    return 0


def get_compatibility_band(score):
    if score >= 85:
        return "High Alignment", "Your values and lifestyle are exceptionally well-matched. You are naturally prepared for long-term stability according to our model."
    if score >= 65:
        return "Growth Partners", "You have a strong foundation to begin with and the right tools as a couple. By addressing a few target areas, you can improve your compatibility as a couple."
    if score >= 40:
        return "Developing Partners", "Your path is unique and occasionally challenging. There are key areas to work on as a couple to build a stronger foundation for a long-term relationship."
    return "Needs Extra Care", "Every great relationship takes effort. You have key growth areas where more conversation and alignment are needed to keep moving forward together."


def get_archetype_profile(
    values_match,
    inlaw_min_band,
    common_int,
    finances,
    household_aligned,
    soulmates,
    age_gap,
    social_app,
    politics,
    premarital,
):
    # Deterministic priority rules (first match wins)
    if values_match >= 2 and finances == "Mostly aligned" and common_int == "Yes" and social_app == "Yes" and inlaw_min_band != "Low":
        return (
            "Power Couple",
            "You are both highly aligned on core life values and shared dynamics – your values, finances, and social lives are in sync.",
            "Avoid complacency: You may unknowingly become complacent about the relationship and take each other for granted, or stop challenging each other to grow.",
            "The good news is you're on the right track! Keep a monthly check-in to discuss important topics, and keep sharing your thoughts and feelings with each other. This ensures both of you continue to stay on the same page without compromising honesty for harmony. Don't forget to have fun together and keep doing the things you both enjoy!",
        )

    if values_match >= 2 and inlaw_min_band == "Low":
        return (
            "Island Tribe",
            "You have a strong bond as a couple, aligning on your shared values which you both cherish and see as important. The 'big picture' matters to you - and this is likely a powerful glue for your relationship.",
            "Friction from family or future in-laws can spill into your relationship and drain your energy. Whether it's on one side or both, it can create tension that may be hard to resolve without support.",
            "Family relationships can be complex and challenging. It may help to create a boundary plan for family events and interactions (e.g. duration, exit cue, and debrief after). Focus on being united when it comes to family, and if possible, work on improving the relationship with your potential in-laws.",
        )

    if common_int == "Yes" and values_match <= 1:
        return (
            "Dynamic Explorers",
            "You connect through shared activities and energy, and your relationship is likely filled with adventure. You enjoy spending time with each other, and your hobbies feel even better when done together!",
            "Without meaning to, topics about the future (kids, money, long-term goals) may be under-discussed or forgotten in the excitement of your relationship.",
            "Deep dive: Build on your relationship by discussing what matters most to you both, and understand each other's perspectives. While both of you may not be certain of what you want, organising a monthly 'State of Us' to talk about the big issues can help you stay aligned and engaged about your future together. Remember to approach these conversations with curiosity and openness (e.g. 'Help me understand why this is important to you?' or 'I see this differently, can I share my perspective?') and recognise that everyone has different values and perspectives.",
        )

    if finances == "Mostly aligned" and household_aligned and soulmates != "Yes - we both do":
        return (
            "Practical Architects",
            "You are a strong operational 'Dream Team' for real-world decisions, and you have the potential to build a well-aligned life together with your shared practical approach.",
            "If you spend most of your time discussing logistics, your relationship may start to become too administrative without emotional replenishment.",
            "Don't forget to have fun! Schedule time for romance and shared hobbies that aren't just about maintaining the household. Consider asking each other 'What can I do to make you feel more loved and appreciated this week?' to strengthen your emotional connection as a couple.",
        )

    if abs(age_gap) >= 5 and social_app == "Yes" and politics != "We share the same views":
        return (
            "Fire & Ice Duo",
            "Different backgrounds coexisting with solid support. Even with a larger age gap, you share a strong social bond.",
            "Differences in values and viewpoints may intensify conflict for couples who lack conflict resolution and repair skills.",
            "Explore different viewpoints respectfully and take breaks during arguments when needed. Try to define 3 non-negotiable shared values to anchor disagreements, and remember to validate each other's feelings and perspectives even when you disagree (e.g. 'I see why that would be important to you' or 'I understand that this is hard for you'). Repairing after conflict is especially important, such as spending quality time together or doing a shared activity you both enjoy.",
        )

    if premarital == "Yes" and inlaw_min_band != "Low" and finances != "Mostly aligned":
        return (
            "Safety Net Couple",
            "You have good support structures as a couple and a willingness to work on the relationship. By taking the step to attend premarital counselling, you have taken a proactive approach to building a strong foundation.",
            "Financial alignment is the main sticking point to work on together.",
            "Try using a simple shared system (rules for saving, spending, and emergency decisions). Discuss your money values and triggers to understand each other's perspectives better (e.g. why one of you prefers to save while the other prefers to spend) and find a common ground that respects both approaches. Since money can be a sensitive topic, remember to approach it with empathy and a problem-solving mindset rather than judgment. It might be helpful to frame it as 'How can we create a money system that works for both of us and reduces stress?'.",
        )

    return (
        "Balanced Builders",
        "You are 'steady growers' as a couple - showing mixed strengths with room to grow into a very stable pattern.",
        "Undefined friction can quietly accumulate without structure or check-ins.",
        "Pick one high-impact area to focus on for 30 days (money alignment, family relationships, or shared hobbies).",
    )


def get_secondary_archetype_hint(values_match, inlaw_min_band, common_int, finances):
    if inlaw_min_band == "Low":
        return "Island Tribe"
    if common_int == "Yes" and values_match <= 1:
        return "Dynamic Explorers"
    if finances == "Mostly aligned":
        return "Practical Architects"
    return "Growth Partners"


def get_archetype_image_path(archetype):
    image_map = {
        "Power Couple": "power_couple",
        "Island Tribe": "island_tribe",
        "Dynamic Explorers": "dynamic_explorers",
        "Practical Architects": "practical_architects",
        "Fire & Ice Duo": "fire_ice_duo",
        "Safety Net Couple": "safety_net_couple",
        "Balanced Builders": "balanced_builders",
    }

    stem = image_map.get(archetype, "balanced_builders")
    images_dir = Path(__file__).resolve().parent / "images"

    png_path = images_dir / f"{stem}.png"
    if png_path.exists():
        return png_path

    svg_path = images_dir / f"{stem}.svg"
    if svg_path.exists():
        return svg_path

    return None


quiz_complete = (
    st.session_state.partner_answers["Girlfriend"] is not None
    and st.session_state.partner_answers["Boyfriend"] is not None
)

if not quiz_complete:
    st.subheader("Player setup")
    active_partner = st.radio(
        "",
        ["Girlfriend", "Boyfriend"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if st.button("Reset both partners' answers"):
        st.session_state.partner_answers = {"Girlfriend": None, "Boyfriend": None}
        st.rerun()

    saved_gf = st.session_state.partner_answers["Girlfriend"] is not None
    saved_bf = st.session_state.partner_answers["Boyfriend"] is not None

    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.info(f"Girlfriend: {'✅ Saved' if saved_gf else '⌛ Not yet saved'}")
    with status_col2:
        st.info(f"Boyfriend: {'✅ Saved' if saved_bf else '⌛ Not yet saved'}")

    st.divider()
    st.subheader("Couple details")
    st.caption("These 3 questions are shared and only need to be answered once.")
    shared_meet_raw = st.selectbox(
        "How did you meet?",
        [
            "School",
            "Workplace",
            "Mutual friends",
            "Online (e.g. Dating App)",
            "Family introduction",
            "Social event/Hobby group",
            "Others",
        ],
        key="shared_meet_raw",
    )
    shared_rel_len = st.selectbox(
        "Relationship length",
        ["<6 months", "6 months–1 year", "1–2 years", "2–4 years", "4+ years"],
        key="shared_rel_len",
    )
    shared_premarital = st.radio(
        "Have you and your partner attended premarital counselling?",
        ["Yes", "No"],
        key="shared_premarital",
    )
   

    # 3. Partner form (hidden after save)
    if st.session_state.partner_answers[active_partner] is None:
        with st.form(f"quiz_{active_partner}"):
            st.header(f"{active_partner} — Your turn")
            st.caption("Your answers are saved and hidden after submission.")

            st.subheader("1) Background")

            col1, col2 = st.columns(2)
            with col1:
                age_at_marriage = st.number_input(
                    "How old will you be when you get married? (If unsure, put your current age)",
                    18,
                    70,
                    28,
                )
                your_parent_status = st.selectbox(
                    "Your parents' relationship status",
                    ["Parents are Married", "Parents are Divorced"],
                )

            with col2:
                soulmate_belief = st.radio(
                    "Pick the statement that feels most true to you",
                    [
                        "What is meant to be for you will always be for you",
                        "Love is built through effort and is always a choice",
                    ],
                )
                household = st.radio("Your ideal household will be", ["Dual income", "Single income"])

            st.divider()
            st.subheader("2) Values & Life Plans")
            children_view = st.selectbox(
                "When it comes to kids",
                ["I want children", "I don't want children", "I'm undecided"],
            )
            politics = st.selectbox(
                "When politics comes up...",
                [
                    "My partner and I see eye to eye",
                    "My partner and I enjoy debating and will agree to disagree",
                    "We avoid the topic altogether",
                    "It causes friction or an argument",
                ],
            )
            finance_windfall = st.selectbox(
                "You and your partner just won S$10,000 in a lucky draw. What feels most like your instinct?",
                FINANCE_WINDFALL_OPTIONS,
            )
            finance_persona = st.selectbox(
                "Which money persona sounds most like you?",
                FINANCE_PERSONA_OPTIONS,
            )
            finance_overspend = st.selectbox(
                "At month-end, expenses were 20% above plan. What would you most likely do?",
                FINANCE_OVERSPEND_OPTIONS,
            )
            values = [
                "Career Ambition",
                "Family",
                "Faith and Spirituality",
                "Adventure/Trying New Things",
                "Security/Stability",
                "Self-Improvement",
                "Travel",
                "Health and Fitness",
                "Community",
            ]
            top3_values = st.multiselect(
                "Which top 3 values do you prioritise most in your life?",
                values,
                max_selections=3,
            )

            st.divider()
            st.subheader("3) Family & Friends")
            inlaw_holiday = st.selectbox(
                "It's a public holiday and your partner's parents expect you to visit, but you value your time. How do you feel?",
                [
                    "I'm happy to go and feel great",
                    "I'm neutral or indifferent",
                    "I rather not go and feel uncomfortable",
                ],
            )
            inlaw_support = st.selectbox(
                "Your partner asks you to check in with their parent this week. What's your likely response?",
                [
                    "Do it willingly and happily",
                    "Do it if needed",
                    "Avoid it if possible",
                ],
            )

            social_invite = st.selectbox(
                "Your partner's close friends invite both of you out. You usually feel...",
                ["Happy and comfortable", "Okay, depends on context and the friends", "Reluctant or drained"],
            )
            social_feeling = st.selectbox(
                "How do you feel about your partner's friends?",
                ["I like them", "I only like some of them", "I don't really like them"],
            )

            st.divider()
            st.subheader("4) Interests & Hobbies")
            common_convo = st.selectbox(
                "After a long day, your conversations are usually...",
                [
                    "We have recurring topics we both get excited about",
                    "Sometimes we have shared topics, sometimes we don't",
                    "We don't have shared topics but have separate things we like to talk about",
                ],
            )
            interest_options = [
                "Fitness/Physical Activity",
                "Eating Out and/or Shopping",
                "Watching Movies/TV",
                "Creative (Music/Art/Crafts)",
                "Spending Time with Family",
                "Volunteering",
                "Intellectual (Reading/Problem-Solving)",
            ]
            common_activities = st.multiselect(
                "Choose your top 2 activities",
                interest_options,
                max_selections=2,
            )
            common_weekend = st.selectbox(
                "It's a free Saturday. What usually happens?",
                [
                    "We naturally choose an activity we both enjoy",
                    "We can find overlap with some effort",
                    "We do our own things",
                ],
            )

            submitted_partner = st.form_submit_button(f"Save {active_partner} responses")

            if submitted_partner:
                if len(top3_values) != 3:
                    st.error("Please choose exactly 3 values.")
                elif len(common_activities) != 2:
                    st.error("Please select exactly 2 activities.")
                else:
                    st.session_state.partner_answers[active_partner] = {
                        "age_at_marriage": age_at_marriage,
                        "your_parent_status": your_parent_status,
                        "meet_raw": shared_meet_raw,
                        "rel_len": shared_rel_len,
                        "premarital": shared_premarital,
                        "soulmate_belief": soulmate_belief,
                        "household": household,
                        "children_view": children_view,
                        "politics": politics,
                        "finance_windfall": finance_windfall,
                        "finance_persona": finance_persona,
                        "finance_overspend": finance_overspend,
                        "inlaw_holiday": inlaw_holiday,
                        "inlaw_support": inlaw_support,
                        "common_weekend": common_weekend,
                        "common_convo": common_convo,
                        "common_activities": common_activities,
                        "social_invite": social_invite,
                        "social_feeling": social_feeling,
                        "top3_values": top3_values,
                    }
                    st.success(f"{active_partner} responses saved and hidden.")
                    st.rerun()
    else:
        st.success(f"{active_partner} responses already saved and hidden.")


# 4. Combined prediction once both partners are saved
if st.session_state.partner_answers["Girlfriend"] and st.session_state.partner_answers["Boyfriend"]:
    st.subheader("Results")
    if st.button("Start new quiz (reset both responses)", type="primary"):
        st.session_state.partner_answers = {"Girlfriend": None, "Boyfriend": None}
        st.rerun()

    gf = st.session_state.partner_answers["Girlfriend"]
    bf = st.session_state.partner_answers["Boyfriend"]

    # Replicate husband-wife orientation in model: Boyfriend - Girlfriend
    age_gap = bf["age_at_marriage"] - gf["age_at_marriage"]
    fam_bg = combine_family_background(gf["your_parent_status"], bf["your_parent_status"])
    meet_clean = combine_meet_clean(gf["meet_raw"], bf["meet_raw"])
    rel_len = combine_relationship_length(gf["rel_len"], bf["rel_len"])
    premarital = "Yes" if gf["premarital"] == "Yes" and bf["premarital"] == "Yes" else "No"
    soulmates = combine_soulmates(gf["soulmate_belief"], bf["soulmate_belief"])
    child_views = combine_children(gf["children_view"], bf["children_view"])
    politics = combine_politics(gf["politics"], bf["politics"])
    gf_inlaw_score = inlaw_relationship_score(gf["inlaw_holiday"], gf["inlaw_support"])
    bf_inlaw_score = inlaw_relationship_score(bf["inlaw_holiday"], bf["inlaw_support"])
    gf_inlaw_label = inlaw_score_to_label(gf_inlaw_score)
    bf_inlaw_label = inlaw_score_to_label(bf_inlaw_score)
    inlaw_min_band = inlaw_band(gf_inlaw_label, bf_inlaw_label)
    household, household_aligned = combine_household(gf["household"], bf["household"])
    a_finance_score = finance_style_score(
        gf["finance_windfall"],
        gf["finance_persona"],
        gf["finance_overspend"],
    )
    b_finance_score = finance_style_score(
        bf["finance_windfall"],
        bf["finance_persona"],
        bf["finance_overspend"],
    )
    a_finance_score, b_finance_score = apply_money_persona_alignment(
        a_finance_score,
        b_finance_score,
        gf["finance_persona"],
        bf["finance_persona"],
    )
    full_money_aligned = is_full_money_alignment(gf, bf)
    finances = combine_finances(
        a_finance_score,
        b_finance_score,
        household_aligned=household_aligned,
        full_money_aligned=full_money_aligned,
    )
    activity_overlap_count = len(set(gf["common_activities"]).intersection(set(bf["common_activities"])))
    a_common_score = common_interest_score(
        gf["common_weekend"],
        gf["common_convo"],
        activity_overlap_count,
    )
    b_common_score = common_interest_score(
        bf["common_weekend"],
        bf["common_convo"],
        activity_overlap_count,
    )
    common_int = combine_common_interest(a_common_score, b_common_score, activity_overlap_count)
    a_social_score = social_circle_score(
        gf["social_invite"],
        gf["social_feeling"],
    )
    b_social_score = social_circle_score(
        bf["social_invite"],
        bf["social_feeling"],
    )
    social_app = combine_social_circle(a_social_score, b_social_score)
    values_match = len(set(gf["top3_values"]).intersection(set(bf["top3_values"])))

    input_data = pd.DataFrame(
        [[
            age_gap,
            fam_bg,
            meet_clean,
            rel_len,
            premarital,
            soulmates,
            child_views,
            politics,
            inlaw_min_band,
            household,
            finances,
            common_int,
            social_app,
            values_match,
        ]],
        columns=RUN4_FEATURES,
    )

    for col in input_data.columns:
        if input_data[col].dtype == "object":
            input_data[col] = input_data[col].astype(str)

    input_t = pre4.transform(input_data)
    input_t = input_t.toarray() if hasattr(input_t, "toarray") else input_t

    expected_n_features = len(model.feature_names_)
    if input_t.shape[1] != expected_n_features:
        st.error(
            f"Model expects {expected_n_features} transformed features, but got {input_t.shape[1]}. "
            "Please re-export the model and preprocessing artifacts from the same training run."
        )
        st.stop()

    prediction = model.predict(input_t)[0]
    probs = model.predict_proba(input_t)[0]
    stable_idx = get_stable_class_index(model.classes_)
    compatibility_score = float(probs[stable_idx] * 100)

    band_name, band_desc = get_compatibility_band(compatibility_score)
    archetype, vibe, risk_note, growth_tip = get_archetype_profile(
        values_match=values_match,
        inlaw_min_band=inlaw_min_band,
        common_int=common_int,
        finances=finances,
        household_aligned=household_aligned,
        soulmates=soulmates,
        age_gap=age_gap,
        social_app=social_app,
        politics=politics,
        premarital=premarital,
    )

    st.divider()
    st.subheader("Compatibility result")
    st.caption(
        "The result reflects patterns from our data set. "
        "It is not a clinical diagnosis or deterministic prediction of your relationship."
    )

    if compatibility_score >= 65:
        st.balloons()
        st.success(f"### {compatibility_score:.1f}% Match · {band_name}")
    elif compatibility_score >= 40:
        st.warning(f"### {compatibility_score:.1f}% Match · {band_name}")
    else:
        st.error(f"### {compatibility_score:.1f}% Match · {band_name}")

    st.write(band_desc)
    st.markdown(f"#### Archetype: **{archetype}**")

    archetype_image = get_archetype_image_path(archetype)
    if archetype_image is not None:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(str(archetype_image), use_container_width=True)

    st.write(f"**The Vibe:** {vibe}")
    st.write(f"**Watch-Out-For:** {risk_note}")
    st.info(f"🎯 **Growth Quest:** {growth_tip}")

    # Mitigation for boundary instability: show a secondary hint near band edges.
    nearest_edge_dist = min(abs(compatibility_score - x) for x in [40, 65, 85])
    if nearest_edge_dist <= 3:
        secondary = get_secondary_archetype_hint(values_match, inlaw_min_band, common_int, finances)
        st.caption(f"Near a band boundary: secondary possible dynamic is **{secondary}**.")

    # Mitigations for interpretation risks
    with st.expander("How to interpret your results responsibly"):
        st.markdown(
            "- **Not deterministic:** Treat this as a tool for self-reflection rather than a verdict. Results may not generalise to all couples. Regardless of compatibility score, the Growth Quest can provide useful next steps for you and your partner to improve your relationship. \n"
        )

    with st.expander("See combined feature values used for prediction"):
        st.dataframe(input_data, use_container_width=True)
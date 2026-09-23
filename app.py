import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="DopaScore | AI Productivity Risk Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 1.4rem;
        border-radius: 9999px;
        font-size: 1.3rem;
        font-weight: 700;
        color: white;
        margin-top: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .badge-none { background: linear-gradient(135deg, #10b981, #059669); }
    .badge-early { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .badge-moderate { background: linear-gradient(135deg, #f97316, #ea580c); }
    .badge-severe { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .badge-critical { background: linear-gradient(135deg, #8b5cf6, #6d28d9); }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_pipeline():
    model_path = 'dopascore_pipeline.joblib'
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), 'dopascore_pipeline.joblib')
    return joblib.load(model_path)

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

# App Header
st.markdown('<div class="main-title">🧠 DopaScore: Productivity Risk Band Predictor</div>', unsafe_allow_html=True)

if not model_loaded:
    st.error(f"Unable to load trained pipeline (`dopascore_pipeline.joblib`). Error: {load_error}")
    st.info("Please run the notebook `Suhani_DopaScore.ipynb` first to train and export the pipeline.")
    st.stop()

# Sidebar: Project Info & Guidance
with st.sidebar:
    st.header("About DopaScore")
    st.markdown("""
    **DopaScore** assesses your vulnerability to digital dopamine dysregulation and sustained productivity impairment.
    
    - 🟢 **None/Minimal**: Healthy digital resilience
    - 🟡 **Early**: Mild distraction & habit friction
    - 🟠 **Moderate**: Noticeable focus fragmentation
    - 🔴 **Severe**: Significant cognitive fatigue
    - 🟣 **Critical**: Severe executive dysfunction & collapse
    """)
    st.divider()
    st.markdown("💡 **Tip:** Adjust the sliders in the form to simulate behavioral interventions and observe risk reduction in real-time.")

# Input Form
st.markdown("### Enter Your Digital & Focus Behavior Profile")

col1, col2 = st.columns(2)

with col1:
    with st.expander("📱 1. Digital & Social Media Consumption", expanded=True):
        avg_daily_sm_hours = st.slider("Daily Social Media Time (Hours)", min_value=0.5, max_value=14.0, value=3.5, step=0.5)
        avg_daily_screen_time_hours = st.slider("Total Daily Screen Time (Hours)", min_value=1.0, max_value=18.0, value=7.0, step=0.5)
        daily_check_frequency = st.selectbox(
            "How frequently do you check your phone?",
            options=['Once a week', 'Once a day', '2-3 times a day', 'Every few hours', 'Constantly'],
            index=3
        )
        first_check_of_day = st.selectbox(
            "When is your first phone check after waking?",
            options=['During morning routine', 'Within 1 hour', 'Within 30 mins', 'Within 5 mins of waking'],
            index=2
        )
        late_night_scrolling = st.selectbox(
            "Late Night Scrolling Frequency (in bed)",
            options=['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'],
            index=2
        )
        doomscrolling_frequency = st.selectbox(
            "Doomscrolling Frequency (negative feed consumption)",
            options=['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'],
            index=2
        )
        notifications_always_on = st.radio(
            "Are notifications always turned on?",
            options=['No', 'Yes'],
            index=1,
            horizontal=True
        )

    with st.expander("⚡ 2. Dopamine Reflexes & Triggers", expanded=True):
        dopamine_rush_feel_score = st.slider("Dopamine Rush / Excitement on Notification (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
        fomo_score = st.slider("Fear Of Missing Out (FOMO) Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)
        validation_seeking_score = st.slider("Validation Seeking Score (Likes/Comments) (1-10)", min_value=1.0, max_value=10.0, value=4.0, step=1.0)
        comparison_to_others_score = st.slider("Social Comparison Feeling (1-10)", min_value=1.0, max_value=10.0, value=4.0, step=1.0)
        boredom_to_phone_reflex = st.selectbox(
            "Do you reflexively reach for your phone whenever bored?",
            options=['Never', 'Rarely', 'Sometimes', 'Yes'],
            index=2
        )
        inability_to_delay_gratification = st.selectbox(
            "Difficulty delaying immediate digital gratification?",
            options=['Never', 'Rarely', 'Sometimes', 'Yes'],
            index=2
        )
        anxiety_when_phone_unavailable = st.slider("Anxiety / Nomophobia when phone unavailable (1-10)", min_value=1.0, max_value=10.0, value=4.0, step=1.0)

with col2:
    with st.expander("🎯 3. Cognitive Endurance & Focus Metrics", expanded=True):
        deep_work_duration_minutes = st.slider("Max Deep Work Block without Disruption (Minutes)", min_value=5.0, max_value=300.0, value=90.0, step=5.0)
        attention_span_minutes = st.slider("Estimated Sustained Attention Span (Minutes)", min_value=3.0, max_value=60.0, value=25.0, step=1.0)
        reading_duration_minutes_per_day = st.slider("Daily Long-Form Reading Duration (Minutes)", min_value=0.0, max_value=180.0, value=30.0, step=5.0)
        task_switching_frequency = st.selectbox(
            "Task Switching Frequency during work/study",
            options=['Low', 'Moderate', 'High'],
            index=1
        )
        difficulty_maintaining_focus = st.selectbox(
            "Difficulty maintaining focus on complex tasks?",
            options=['Never', 'Rarely', 'Sometimes', 'Yes'],
            index=2
        )
        difficulty_starting_tasks = st.selectbox(
            "Difficulty initiating procrastinated tasks?",
            options=['Never', 'Rarely', 'Sometimes', 'Yes'],
            index=2
        )
        mental_fog_frequency = st.selectbox(
            "Mental Fog / Cognitive Fatigue Frequency",
            options=['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'],
            index=2
        )
        decision_fatigue_score = st.slider("Decision Fatigue Score at End of Day (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=1.0)

    with st.expander("🌿 4. Rest, Lifestyle & Demographics", expanded=True):
        avg_sleep_hours = st.slider("Average Sleep Duration (Hours/Night)", min_value=3.0, max_value=10.0, value=7.0, step=0.5)
        sleep_quality = st.selectbox(
            "Subjective Sleep Quality",
            options=['Very poor', 'Poor', 'Fair', 'Good', 'Very good'],
            index=3
        )
        exercise_frequency = st.selectbox(
            "Weekly Physical Exercise Frequency",
            options=['Never', '1-2x/week', '3-4x/week', 'Daily'],
            index=2
        )
        age = st.number_input("Age", min_value=13, max_value=85, value=22)
        education_level = st.selectbox("Education Level", ['High school', 'Undergraduate', 'Postgraduate'], index=1)
        employment_status = st.selectbox("Employment Status", ['Student', 'Full-time employed', 'Part-time employed', 'Job seeker', 'Retired'], index=0)

# Build full 54-feature record for the pipeline
def construct_feature_row():
    if age <= 17:
        ag = '13-17'
    elif age <= 22:
        ag = '18-22'
    elif age <= 27:
        ag = '23-27'
    elif age <= 35:
        ag = '28-35'
    elif age <= 49:
        ag = '36-49'
    elif age <= 64:
        ag = '50-64'
    else:
        ag = '65+'

    row = {
        'age': float(age),
        'age_group': ag,
        'gender': 'Female',
        'race_ethnicity': 'South Asian',
        'nationality': 'Indian',
        'country_of_residence': 'India',
        'education_level': education_level,
        'employment_status': employment_status,
        'occupation_category': 'Education' if employment_status == 'Student' else 'Technology',
        'relationship_status': 'Single',
        'avg_daily_sm_hours': float(avg_daily_sm_hours),
        'years_on_social_media': 5.0,
        'primary_platform': 'Instagram',
        'secondary_platform': 'YouTube',
        'daily_check_frequency': daily_check_frequency,
        'first_check_of_day': first_check_of_day,
        'late_night_scrolling': late_night_scrolling,
        'sm_during_work_study': 'Sometimes',
        'sm_during_meals': 'Sometimes',
        'notifications_always_on': notifications_always_on,
        'scroll_without_purpose': 'Sometimes',
        'doomscrolling_frequency': doomscrolling_frequency,
        'content_type_consumed': 'Short video/Reels',
        'avg_daily_screen_time_hours': float(avg_daily_screen_time_hours),
        'avg_sleep_hours': float(avg_sleep_hours),
        'sleep_quality': sleep_quality,
        'exercise_frequency': exercise_frequency,
        'avg_work_study_hours_per_day': 6.0,
        'dopamine_rush_feel_score': float(dopamine_rush_feel_score),
        'reward_seeking_behavior': 3.0,
        'fomo_score': float(fomo_score),
        'comparison_to_others_score': float(comparison_to_others_score),
        'validation_seeking_score': float(validation_seeking_score),
        'boredom_to_phone_reflex': boredom_to_phone_reflex,
        'inability_to_delay_gratification': inability_to_delay_gratification,
        'attention_span_minutes': float(attention_span_minutes),
        'deep_work_duration_minutes': float(deep_work_duration_minutes),
        'task_switching_frequency': task_switching_frequency,
        'difficulty_starting_tasks': difficulty_starting_tasks,
        'difficulty_maintaining_focus': difficulty_maintaining_focus,
        'mind_wandering_during_work': 'Sometimes',
        'creative_output_frequency': 'Moderate',
        'reading_duration_minutes_per_day': float(reading_duration_minutes_per_day),
        'memory_recall_difficulty': 'Sometimes',
        'learning_retention_difficulty': 'Sometimes',
        'decision_fatigue_score': float(decision_fatigue_score),
        'mental_fog_frequency': mental_fog_frequency,
        'irritability_when_offline': 'Sometimes',
        'anxiety_when_phone_unavailable': float(anxiety_when_phone_unavailable),
        'restlessness_score': float(anxiety_when_phone_unavailable * 0.9),
        'mood_dependency_on_sm': 'Sometimes',
        'withdrawal_attempt_count': 3.0,
        'longest_detox_days': 4.0,
        'detox_relapse_reason': 5.0
    }
    return pd.DataFrame([row])

st.divider()

# Predict Button & Prediction Results Section
predict_btn = st.button("🔍 Assess My Productivity Risk Band", type="primary", use_container_width=True)

if predict_btn:
    input_df = construct_feature_row()
    
    with st.spinner("Analyzing behavioral vectors through trained pipeline..."):
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        prob_dict = dict(zip(classes, probabilities))
    
    st.subheader("Diagnostic Results")
    
    badge_map = {
        'None/Minimal': ('badge-none', '🟢 None / Minimal Risk', 'Cognitive resilience is high. Healthy digital boundaries protect deep work capacity.'),
        'Early': ('badge-early', '🟡 Early Risk Band', 'Early indicators of attention drift and phone reflex. Preventative action recommended.'),
        'Moderate': ('badge-moderate', '🟠 Moderate Risk Band', 'Noticeable focus fragmentation. Compulsive habit loops are reducing sustained productivity.'),
        'Severe': ('badge-severe', '🔴 Severe Risk Band', 'Acute digital dopamine dysregulation. Deep work is heavily curtailed by frequent task-switching.'),
        'Critical': ('badge-critical', '🟣 Critical Risk Band', 'Critical attention collapse. Digital hyper-stimulation is inducing severe chronic productivity failure.')
    }
    
    badge_class, badge_label, badge_desc = badge_map.get(prediction, ('badge-moderate', prediction, ''))
    
    res_col1, res_col2 = st.columns([1.2, 1.8])
    
    with res_col1:
        st.markdown(f'<div class="risk-badge {badge_class}">{badge_label}</div>', unsafe_allow_html=True)
        st.write(badge_desc)
        st.metric("Confidence in Diagnosis", f"{prob_dict[prediction]*100:.1f}%")
    
    with res_col2:
        st.markdown("**Predicted Probability Distribution Across Risk Bands:**")
        for band in ['None/Minimal', 'Early', 'Moderate', 'Severe', 'Critical']:
            prob = prob_dict.get(band, 0.0)
            st.write(f"**{band}**: {prob*100:.1f}%")
            st.progress(float(prob))

    # Key Behavioral Contributing Factors
    st.markdown("### 🔍 Key Behavioral Contributing Factors for This Profile")
    factors = []
    if avg_daily_sm_hours > 5.0:
        factors.append(f"⚠️ **High Social Media Exposure**: {avg_daily_sm_hours} hours/day exceeds the healthy cognitive threshold (<3.0 hrs).")
    if deep_work_duration_minutes < 45.0:
        factors.append(f"⚠️ **Compressed Deep Work**: Sustained deep work ({deep_work_duration_minutes} mins) is severely compromised by digital friction.")
    if reading_duration_minutes_per_day < 20.0:
        factors.append(f"⚠️ **Reduced Linear Focus**: Only {reading_duration_minutes_per_day} minutes of daily reading weakens sustained mental stamina.")
    if fomo_score >= 7.0:
        factors.append(f"⚠️ **Elevated FOMO Reaction**: Score of {fomo_score}/10 triggers anxiety and compulsive checking reflex.")
    if late_night_scrolling in ['Often', 'Daily']:
        factors.append("⚠️ **Nocturnal Screen Usage**: Bedtime scrolling impairs melatonin synthesis and REM sleep consolidation.")
    if avg_sleep_hours < 6.5:
        factors.append(f"⚠️ **Sleep Deprivation**: {avg_sleep_hours} hours/night directly degrades prefrontal executive function.")
        
    if not factors:
        st.success("✅ Your digital habits are well-balanced! All critical markers remain within protective thresholds.")
    else:
        for factor in factors:
            st.markdown(factor)

    # Actionable Protocol Recommendations
    st.markdown("### 📋 Tailored Digital Hygiene Protocol")
    if prediction in ['None/Minimal', 'Early']:
        st.info("""
        - **Maintain Boundaries:** Protect your morning routine by delaying phone access for 30 minutes after waking.
        - **Deep Work Blocks:** Schedule 2x 60-minute uninterrupted deep focus blocks daily with notifications muted.
        - **Greyscale Display:** Enable greyscale mode in evening hours to suppress dopamine stimulation.
        """)
    else:
        st.warning("""
        - **Digital Sunset:** Place your phone outside the bedroom 60 minutes before scheduled sleep.
        - **App Pruning & Notification Audit:** Disable all non-human notifications (news, algorithmic recommendations, social alerts).
        - **Dopamine Reset Sprint:** Implement the Pomodoro technique (25 min single-tasking / 5 min walk without phone).
        - **Restore Linear Focus:** Commit to 20 minutes of physical book reading daily to recondition long-form cognitive endurance.
        """)

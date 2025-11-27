import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LinearRegression

# ======================================================================
# 🛠️ 1. حل مشكلة الاستيراد (Smart Import)
# ======================================================================
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            st.error("❌ خطأ: مكتبة TensorFlow غير مثبتة.")
            st.stop()

# ======================================================================
# -------------------- 2. تحميل الأصول والبيانات --------------------
# ======================================================================
@st.cache_resource
def load_assets_lite():
    # القواميس
    recommendations_map = {
        "الكفاءة للعنصر البشري": "تطوير برامج تدريبية مستمرة للمعلمين وربطها بتقييم الأداء الفردي.",
        "المناهج": "مراجعة شاملة للمناهج وتحديثها لتتوافق مع مهارات القرن 21.",
        "التطور المهني": "إنشاء مسارات مهنية واضحة للمعلمين مع حوافز مرتبطة بالإنجاز.",
        "تعزيز الشخصية": "إطلاق برامج إرشاد نفسي واجتماعي لتعزيز الثقة والقيادة لدى الطلاب.",
        "التقويم التربوي": "تطوير أدوات تقييم معيارية رقمية لقياس نواتج التعلم بدقة.",
        "الشراكة مع القطاع الخاص": "توسيع الشراكات مع الشركات لدعم التدريب العملي والموارد التقنية.",
        "مشاركة الاسرة": "تفعيل مجالس أولياء الأمور وإشراكهم في متابعة الأداء المدرسي.",
        "المرافق التعليمية والمباني": "تحديث البنية التحتية وتوفير بيئة صفية جاذبة وآمنة.",
        "التقنية بالمدارس": "تعميم الفصول الذكية وربطها بمنصات تعليمية رقمية.",
        "قياس الأداء المدرسي": "إدخال مؤشرات أداء رئيسية (KPIs) لمتابعة تقدم المدارس بشكل دوري.",
        "استراتيجيات التدريس": "تطبيق استراتيجيات تعلم نشط وتدريس تفريدية تراعي الفروق الفردية.",
        "الاختبارات المعيارية": "إعداد اختبارات معيارية وطنية لمقارنة الأداء بين المدارس والمناطق."
    }
    
    clusters = {
        "تعليم": {"استراتيجيات التدريس","المناهج","التطور المهني"},
        "تقييم": {"التقويم التربوي","الاختبارات المعيارية","قياس الأداء المدرسي"},
        "أسرة ومجتمع": {"مشاركة الاسرة","الشراكة مع القطاع الخاص","تعزيز الشخصية"},
        "بيئة وتجهيز": {"المرافق التعليمية والمباني","التقنية بالمدارس"}
    }

    try:
        if not os.path.exists('ranking_model_lite.tflite'): return None
        interpreter = Interpreter(model_path='ranking_model_lite.tflite')
        interpreter.allocate_tensors()

        scaler_X = joblib.load('scaler_X_lite.pkl')
        scaler_y = joblib.load('scaler_y_lite.pkl')
        
        indicator_names = []
        if os.path.exists('indicator_names_lite.txt'):
            with open('indicator_names_lite.txt', 'r', encoding='utf-8') as f:
                indicator_names = [line.strip() for line in f]
        
        feature_importance_map = joblib.load('feature_importance_map.pkl') if os.path.exists('feature_importance_map.pkl') else {}
        if not feature_importance_map and indicator_names:
            feature_importance_map = {name: 1.0 for name in indicator_names}

        return interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, clusters, feature_importance_map
    
    except Exception as e:
        return None

loaded_assets = load_assets_lite()
if loaded_assets is None:
    st.error("⚠️ الملفات الأساسية مفقودة. تأكد من رفع ملفات النموذج.")
    st.stop()

interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, clusters, feature_importance_map = loaded_assets

# ======================================================================
# -------------------- 3. دوال التنبؤ والمحاكاة (PARTS Core) --------------------
# ======================================================================

def forecast_future_values(df_history, target_year, indicators):
    """ التنبؤ بقيم المؤشرات لسنة محددة بناءً على البيانات التاريخية """
    row_data = {}
    years_train = df_history['السنة'].values.reshape(-1, 1)
    
    for col in indicators:
        if col in df_history.columns:
            # تدريب نموذج خطي لكل مؤشر لاكتشاف الـ Trend
            model = LinearRegression()
            y_train = df_history[col].values
            model.fit(years_train, y_train)
            predicted_val = model.predict([[target_year]])[0]
            row_data[col] = max(0.0, min(100.0, predicted_val))
        else:
            row_data[col] = 50.0 # قيمة افتراضية
    return row_data

def run_ai_model(input_values_dict, interpreter, scaler_X, scaler_y, indicator_names):
    """ تشغيل نموذج الذكاء الاصطناعي للحصول على الترتيب """
    values_list = [input_values_dict[name] for name in indicator_names]
    input_array = np.array([values_list]).astype(np.float32)
    
    X_scaled = scaler_X.transform(input_array)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    
    rank = scaler_y.inverse_transform(y_scaled).flatten()[0]
    return max(1.0, rank)

def calculate_synergy(current_inputs, indicator_names, clusters):
    """ حساب التآزر والمؤشرات الضعيفة """
    weak_inds = [name for name in indicator_names if current_inputs[name] < 60]
    hits = {c: len(set(weak_inds) & members) for c, members in clusters.items()}
    boost = 1.0 + (sum(1 for v in hits.values() if v >= 2) * 0.08)
    return min(boost, 1.25), weak_inds

# ======================================================================
# -------------------- 4. واجهة المستخدم (PARTS Framework UI) --------------------
# ======================================================================

st.set_page_config(layout="wide", page_title="نظام PARTS الهجين")

st.markdown("""
    <style>
        .main { direction: rtl; }
        .stSlider > div { direction: rtl; }
        h1, h2, h3, p, div { text-align: right; font-family: 'Tahoma'; }
        div[data-testid="stMetricValue"] { direction: rtl; }
        .stTabs [data-baseweb="tab-list"] { justify-content: flex-end; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 نظام التنبؤ والمحاكاة الهجين (Hybrid PARTS Model)")
st.markdown("---")

# --- 1. البيانات والإعدادات ---
st.sidebar.header("📂 1. البيانات التاريخية")
uploaded_file = st.sidebar.file_uploader("ارفع ملف Excel (السنوات السابقة)", type=["xlsx"])

if uploaded_file is not None:
    df_history = pd.read_excel(uploaded_file)
    
    if 'السنة' not in df_history.columns:
        st.error("يجب أن يحتوي الملف على عمود 'السنة'")
        st.stop()
        
    last_year = int(df_history['السنة'].max())
    
    # === التعديل هنا: السماح باختيار عدة سنوات + نطاق مفتوح (10 سنوات) ===
    future_years_options = [last_year + i for i in range(1, 11)]
    selected_years = st.sidebar.multiselect(
        "اختر السنوات المستقبلية للتنبؤ بها:",
        options=future_years_options,
        default=[last_year + 1]
    )
    
    if not selected_years:
        st.warning("الرجاء اختيار سنة واحدة على الأقل.")
        st.stop()

    st.header("📊 النتائج والمحاكاة (PARTS Simulator)")
    
    # إنشاء تبويبات (Tabs) لكل سنة مختارة
    tabs = st.tabs([str(year) for year in selected_years])
    
    for i, target_year in enumerate(selected_years):
        with tabs[i]:
            st.markdown(f"### 🗓️ محاكاة سنة {target_year}")
            
            # 1. التنبؤ الأولي لهذه السنة (Forecast Baseline)
            forecasted_values = forecast_future_values(df_history, target_year, indicator_names)
            
            col_sim, col_results = st.columns([1, 2])
            
            # >> عمود المحاكاة (Sliders)
            with col_sim:
                st.info("🔧 اضبط المؤشرات (Simulation)")
                user_inputs = {}
                for name in indicator_names:
                    default_val = float(forecasted_values[name])
                    # مفتاح فريد لكل سنة لتجنب تداخل السلايدر
                    slider_key = f"{name}_{target_year}"
                    
                    user_inputs[name] = st.slider(
                        f"{name}", 0.0, 100.0, default_val, key=slider_key
                    )
            
            # >> عمود النتائج (Results)
            with col_results:
                # تشغيل النموذج
                current_rank = run_ai_model(user_inputs, interpreter, scaler_X, scaler_y, indicator_names)
                baseline_rank = run_ai_model(forecasted_values, interpreter, scaler_X, scaler_y, indicator_names)
                
                # حساب التآزر
                synergy_factor, weak_inds = calculate_synergy(user_inputs, indicator_names, clusters)
                
                # عرض النتائج
                m1, m2, m3 = st.columns(3)
                m1.metric("الترتيب المتوقع", f"{current_rank:.2f}")
                m2.metric("معامل التآزر", f"{synergy_factor:.2f}x")
                m3.metric("مؤشرات حرجة", f"{len(weak_inds)}")
                
                # الرسم البياني (تم حله باستخدام الأعمدة)
                st.markdown("#### 📈 أثر التدخل على الترتيب")
                chart_data = pd.DataFrame({
                    "التنبؤ الآلي (Baseline)": [baseline_rank],
                    "بعد المحاكاة (Simulation)": [current_rank]
                })
                st.bar_chart(chart_data, color=["#FF5722", "#4CAF50"])
                
                # التوصيات
                st.markdown("#### 💡 التوصيات الذكية")
                if weak_inds:
                    recs = []
                    for ind in weak_inds:
                        recs.append({
                            "المؤشر": ind,
                            "التوصية": recommendations_map.get(ind, "-"),
                            "الأهمية": f"{feature_importance_map.get(ind, 0.5):.2f}"
                        })
                    st.dataframe(pd.DataFrame(recs), use_container_width=True)
                else:
                    st.success("أداء ممتاز! جميع

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
    # القواميس (التوصيات والخطط)
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
    st.error("⚠️ الملفات الأساسية مفقودة. تأكد من رفع ملفات النموذج (.tflite, .pkl).")
    st.stop()

interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, clusters, feature_importance_map = loaded_assets

# ======================================================================
# -------------------- 3. دوال التنبؤ والمحاكاة (PARTS Core) --------------------
# ======================================================================

def forecast_future_values(df_history, target_year, indicators):
    """ التنبؤ (Prediction): الخطوة الأولى في PARTS """
    row_data = {}
    years_train = df_history['السنة'].values.reshape(-1, 1)
    
    for col in indicators:
        if col in df_history.columns:
            model = LinearRegression()
            y_train = df_history[col].values
            model.fit(years_train, y_train)
            predicted_val = model.predict([[target_year]])[0]
            row_data[col] = max(0.0, min(100.0, predicted_val))
        else:
            # إصلاح المشكلة: بدلاً من 0، نضع متوسط (50) لتجنب انهيار النموذج
            row_data[col] = 50.0 
    return row_data

def run_ai_model(input_values_dict, interpreter, scaler_X, scaler_y, indicator_names):
    """ تشغيل النموذج (AI-Driven) """
    values_list = [input_values_dict[name] for name in indicator_names]
    input_array = np.array([values_list]).astype(np.float32)
    
    # 1. التطبيع
    X_scaled = scaler_X.transform(input_array)
    
    # 2. التنبؤ
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    
    # 3. عكس التطبيع
    rank = scaler_y.inverse_transform(y_scaled).flatten()[0]
    
    # إصلاح الأرقام السالبة (الترتيب لا يمكن أن يكون سالباً)
    return max(1.0, rank)

def calculate_synergy(current_inputs, indicator_names, clusters):
    """ حساب التآزر (Synergy) """
    # نعتبر المؤشرات الضعيفة هي التي تقل عن 60%
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
        .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; text-align: center; }
        .highlight { color: #4CAF50; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 نظام التنبؤ والمحاكاة الهجين (Hybrid PARTS Model)")
st.markdown("---")

# --- 1. القسم الأول: البيانات (Data Input) ---
st.sidebar.header("📂 1. البيانات التاريخية")
uploaded_file = st.sidebar.file_uploader("ارفع ملف Excel (السنوات السابقة)", type=["xlsx"])

if uploaded_file is not None:
    df_history = pd.read_excel(uploaded_file)
    
    if 'السنة' not in df_history.columns:
        st.error("يجب أن يحتوي الملف على عمود 'السنة'")
        st.stop()
        
    last_year = int(df_history['السنة'].max())
    target_year = st.sidebar.selectbox("اختر سنة التنبؤ المستهدفة:", [last_year + i for i in range(1, 6)])
    
    # --- التنبؤ الأولي (Forecast Baseline) ---
    forecasted_values = forecast_future_values(df_history, target_year, indicator_names)
    
    # --- 2. القسم الثاني: المحاكاة والتآزر (Simulation & Synergy) ---
    st.header(f"🎛️ لوحة المحاكاة التفاعلية لسنة {target_year}")
    st.info("💡 **المرحلة الهجينة:** القيم أدناه تم التنبؤ بها آلياً (AI Prediction). يمكنك الآن تعديلها يدوياً (Simulation) لرؤية أثر القرارات.")
    
    # تقسيم الشاشة
    col_sim, col_results = st.columns([1, 2])
    
    # >> عمود المحاكاة (Sliders)
    with col_sim:
        st.markdown("### 🔧 ضبط المؤشرات (Simulation)")
        user_inputs = {}
        for name in indicator_names:
            # القيمة الافتراضية هي القيمة المتنبأ بها
            default_val = float(forecasted_values[name])
            user_inputs[name] = st.slider(f"{name}", 0.0, 100.0, default_val, key=name)
            
            # عرض الفرق عن التنبؤ
            diff = user_inputs[name] - default_val
            if diff != 0:
                st.caption(f"تغيير عن التنبؤ: {diff:+.1f}%")

    # >> عمود النتائج (Results)
    with col_results:
        # 1. تشغيل النموذج على القيم الحالية (سواء كانت متنبأ بها أو معدلة)
        current_rank = run_ai_model(user_inputs, interpreter, scaler_X, scaler_y, indicator_names)
        
        # 2. حساب التآزر
        synergy_factor, weak_inds = calculate_synergy(user_inputs, indicator_names, clusters)
        
        st.markdown("### 📊 النتائج والتشخيص (Analysis & Diagnosis)")
        
        # عرض الميتركس
        m1, m2, m3 = st.columns(3)
        m1.metric("الترتيب المتوقع (النتيجة النهائية)", f"{current_rank:.2f}")
        m2.metric("معامل التآزر (Synergy)", f"{synergy_factor:.2f}x")
        m3.metric("عدد المؤشرات الحرجة", f"{len(weak_inds)}")
        
        # الرسم البياني للمقارنة (Baseline vs Simulation)
        st.markdown("#### 📈 أثر المحاكاة على الترتيب")
        
        # نحسب الترتيب "الأساسي" (بدون تعديلات المستخدم) للمقارنة
        baseline_rank = run_ai_model(forecasted_values, interpreter, scaler_X, scaler_y, indicator_names)
        
        chart_data = pd.DataFrame({
            "السيناريو": ["التنبؤ الأصلي (Baseline)", "المحاكاة الحالية (Simulation)"],
            "الترتيب (الأقل أفضل)": [baseline_rank, current_rank]
        })
        st.bar_chart(chart_data.set_index("السيناريو"), color=["#FF5722", "#4CAF50"])
        
        if current_rank < baseline_rank:
            st.success(f"✅ محاكاتك أدت إلى تحسين الترتيب بمقدار {baseline_rank - current_rank:.2f} نقطة!")
        elif current_rank > baseline_rank:
            st.warning(f"⚠️ التعديلات الحالية أدت لتراجع الترتيب بمقدار {current_rank - baseline_rank:.2f} نقطة.")

        # التوصيات (Recommendations)
        st.markdown("### 💡 التوصيات الذكية (Recommendations)")
        if weak_inds:
            recs = []
            for ind in weak_inds:
                recs.append({
                    "المؤشر": ind,
                    "التوصية": recommendations_map.get(ind, "مراجعة الخطة التشغيلية"),
                    "الأهمية": f"{feature_importance_map.get(ind, 0.5):.2f}"
                })
            st.table(pd.DataFrame(recs))
        else:
            st.success("🎉 جميع المؤشرات في وضع ممتاز في هذه المحاكاة!")

else:
    # شاشة الترحيب
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>👋 مرحبًا بك في منصة PARTS الهجينة</h2>
        <p>للبدء، يرجى رفع ملف البيانات التاريخية من القائمة الجانبية.</p>
        <p style='color: gray;'>سيقوم النظام تلقائيًا بالتنبؤ بالمستقبل ثم يتيح لك محاكاة القرارات.</p>
    </div>
    """, unsafe_allow_html=True)

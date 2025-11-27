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
            st.error("❌ خطأ: مكتبة TensorFlow غير مثبتة (تأكد من requirements.txt).")
            st.stop()

# ======================================================================
# -------------------- 2. تحميل الأصول والبيانات --------------------
# ======================================================================

@st.cache_resource
def load_assets_lite():
    # القواميس الثابتة
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
    
    execution_plan_map = {
        "الكفاءة للعنصر البشري": "توزيع برامج تدريبية حسب مستويات المعلمين وربطها بتقييم الأداء السنوي.",
        "المناهج": "تشكيل لجان مراجعة للمناهج وربط التحديثات بنتائج الاختبارات المعيارية.",
        "التطور المهني": "تصميم مسارات مهنية فردية مع متابعة فصلية وتقييم تطبيقي.",
        "تعزيز الشخصية": "تنفيذ أنشطة صفية ولاصفية تعزز القيادة والانضباط الذاتي.",
        "التقويم التربوي": "إعادة تصميم أدوات التقويم وربطها بمؤشرات الأداء المدرسي.",
        "الشراكة مع القطاع الخاص": "توقيع اتفاقيات تعاون مع شركات محلية لدعم التدريب والمرافق.",
        "مشاركة الاسرة": "إطلاق منصة تواصل مع أولياء الأمور وربطها بتقارير الأداء.",
        "المرافق التعليمية والمباني": "تحديد أولويات الصيانة والتجهيز حسب كثافة الطلاب.",
        "التقنية بالمدارس": "توزيع الأجهزة وربطها بمنصات تعليمية وتدريب المعلمين عليها.",
        "قياس الأداء المدرسي": "تطبيق نظام مؤشرات أداء شهري وربطه بالتحفيز الإداري.",
        "استراتيجيات التدريس": "تدريب المعلمين على التعلم النشط والتقويم التكويني.",
        "الاختبارات المعيارية": "تصميم اختبارات وطنية موحدة وربط نتائجها بخطط التحسين."
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
        
        # التأكد من ملء خريطة الأهمية بقيم افتراضية إذا كانت فارغة
        if not feature_importance_map and indicator_names:
            feature_importance_map = {name: 1.0 for name in indicator_names}

        return interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map
    
    except Exception as e:
        st.error(f"خطأ في تحميل الأصول: {e}")
        return None

# تحميل الأصول
loaded_assets = load_assets_lite()
if loaded_assets is None:
    st.error("⚠️ يرجى التأكد من رفع ملفات النموذج (.tflite, .pkl, .txt) بجانب ملف الكود.")
    st.stop()

interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map = loaded_assets

# ======================================================================
# -------------------- 3. منطق التنبؤ المستقبلي (Forecasting) --------------------
# ======================================================================

def forecast_future_values(df_history, target_years, indicators):
    """
    تقوم هذه الدالة بالتنبؤ بقيم المؤشرات للسنوات القادمة بناءً على البيانات التاريخية
    باستخدام الانحدار الخطي (Linear Regression) لكل مؤشر على حدة.
    """
    future_data = []
    
    # التأكد من وجود عمود السنة
    if 'السنة' not in df_history.columns:
        st.error("يجب أن يحتوي ملف الإكسل على عمود باسم 'السنة'")
        return None

    years_train = df_history['السنة'].values.reshape(-1, 1)

    for future_year in target_years:
        row = {'السنة': future_year, 'نوع البيانات': 'متنبأ بها'}
        
        for col in indicators:
            if col in df_history.columns:
                # تدريب نموذج خطي بسيط لاكتشاف الاتجاه (Trend)
                model = LinearRegression()
                y_train = df_history[col].values
                model.fit(years_train, y_train)
                
                # التنبؤ بالقيمة المستقبلية
                predicted_val = model.predict([[future_year]])[0]
                # ضمان أن القيمة منطقية (بين 0 و 100)
                predicted_val = max(0.0, min(100.0, predicted_val))
                
                row[col] = predicted_val
            else:
                row[col] = 0.0 # قيمة افتراضية في حال نقص العمود
        
        future_data.append(row)
    
    return pd.DataFrame(future_data)

# ======================================================================
# -------------------- 4. منطق الذكاء الاصطناعي (Ranking AI) --------------------
# ======================================================================

def run_ai_ranking(input_values, interpreter, scaler_X, scaler_y, indicator_names):
    # تجهيز البيانات للنموذج
    input_array = np.array([input_values]).astype(np.float32)
    X_scaled = scaler_X.transform(input_array)
    
    # TFLite Inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    
    # النتيجة النهائية (الترتيب)
    rank = scaler_y.inverse_transform(y_scaled).flatten()[0]
    return rank

def analyze_year(row_data, indicator_names):
    # استخراج القيم فقط للتحليل
    values = [row_data[col] for col in indicator_names]
    
    # 1. التنبؤ بالترتيب
    rank_pred = run_ai_ranking(values, interpreter, scaler_X, scaler_y, indicator_names)
    
    # 2. تحليل المؤشرات الضعيفة
    # (نقوم بتطبيع البيانات محلياً لمعرفة الأضعف نسبياً)
    scaled_vals = np.array(values) / 100.0 
    risks = sorted(zip(indicator_names, scaled_vals), key=lambda x: x[1])
    top_weak_inds = [r[0] for r in risks[:5]]
    
    # 3. حساب التآزر والمكاسب
    m_synergy = 0
    selected_set = set(top_weak_inds)
    hits = {c: len(selected_set & members) for c, members in clusters.items()}
    same_cluster_boost = sum(1 for _, v in hits.items() if v >= 2) * 0.08
    multi_cluster_boost = sum(1 for _, v in hits.items() if v >= 1) * 0.03
    m_synergy = min(1.0 + same_cluster_boost + multi_cluster_boost, 1.25)
    
    importance_sum = sum([feature_importance_map.get(i, 0.05) for i in top_weak_inds])
    total_gain = rank_pred * 0.1 * importance_sum * m_synergy
    
    return rank_pred, total_gain, m_synergy, top_weak_inds

# ======================================================================
# -------------------- 5. واجهة المستخدم (Streamlit UI) --------------------
# ======================================================================

st.set_page_config(layout="wide", page_title="منصة التنبؤ المدرسي")

# تنسيق عربي
st.markdown("""
    <style>
        .main { direction: rtl; }
        div[data-testid="stFileUploader"] { text-align: right; }
        h1, h2, h3, p, div { text-align: right; }
        .stDataFrame { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 منصة الذكاء الاصطناعي للتنبؤ بترتيب المدارس")
st.markdown("### نظام تنبؤي قائم على البيانات التاريخية (Data-Driven Forecasting)")
st.markdown("---")

# --- الشريط الجانبي: رفع الملف واختيار السنوات ---
st.sidebar.header("📂 البيانات والإعدادات")

uploaded_file = st.sidebar.file_uploader("ارفع ملف Excel (يحتوي على البيانات التاريخية)", type=["xlsx"])

if uploaded_file is not None:
    # قراءة الملف
    df_history = pd.read_excel(uploaded_file)
    
    # التحقق من الأعمدة
    missing_cols = [col for col in indicator_names if col not in df_history.columns]
    if 'السنة' not in df_history.columns:
        st.error("❌ الملف يجب أن يحتوي على عمود 'السنة'.")
        st.stop()
        
    if missing_cols:
        st.warning(f"⚠️ تنبيه: بعض الأعمدة مفقودة في الملف: {missing_cols}. سيتم اعتبار قيمها 0.")

    st.sidebar.success(f"✅ تم تحميل بيانات {len(df_history)} سنوات سابقة.")

    # اختيار سنوات التنبؤ
    last_year = int(df_history['السنة'].max())
    future_years_options = [last_year + i for i in range(1, 6)] # اقترح 5 سنوات قادمة
    selected_years = st.sidebar.multiselect("اختر السنوات المستقبلية للتنبؤ بها:", future_years_options, default=[last_year+1])
    
    if st.sidebar.button("ابدأ التنبؤ والتحليل ⚡", type="primary"):
        
        # 1. التنبؤ بقيم المؤشرات (Forecasting)
        with st.spinner('جاري التنبؤ بقيم المؤشرات المستقبلية وتحليل الترتيب...'):
            df_forecast = forecast_future_values(df_history, selected_years, indicator_names)
            
            if df_forecast is not None:
                st.subheader(f"📅 النتائج التنبؤية للسنوات القادمة ({', '.join(map(str, selected_years))})")
                
                # علامات تبويب للسنوات
                tabs = st.tabs([str(y) for y in selected_years])
                
                for i, year in enumerate(selected_years):
                    with tabs[i]:
                        # بيانات السنة الحالية من التنبؤ
                        current_row = df_forecast[df_forecast['السنة'] == year].iloc[0]
                        
                        # تشغيل نموذج الذكاء الاصطناعي لهذه السنة
                        rank, gain, synergy, weak_inds = analyze_year(current_row, indicator_names)
                        
                        # --- عرض لوحة القيادة (Dashboard) ---
                        c1, c2, c3 = st.columns(3)
                        c1.metric("الترتيب المتنبأ به", f"{rank:.2f}")
                        c2.metric("مكسب التحسن المتوقع", f"+{gain:.2f}", f"تآزر: {synergy:.2f}x")
                        c3.metric("عدد المؤشرات الحرجة", f"{len(weak_inds)} مؤشرات")
                        
                        st.markdown("#### 📉 قيم المؤشرات المتنبأ بها لهذه السنة")
                        # عرض المؤشرات كـ Progress Bars
                        col_ind1, col_ind2 = st.columns(2)
                        for idx, name in enumerate(indicator_names):
                            val = current_row[name]
                            with (col_ind1 if idx % 2 == 0 else col_ind2):
                                st.write(f"**{name}**: {val:.1f}")
                                st.progress(int(val))

                        st.markdown("---")
                        st.markdown("#### 📝 التوصيات الذكية وخطط العمل")
                        
                        recs_data = []
                        for ind in weak_inds:
                            recs_data.append({
                                "المؤشر": ind,
                                "الأهمية النسبية": f"{feature_importance_map.get(ind, 0):.2f}",
                                "التوصية": recommendations_map.get(ind, "-"),
                                "خطة التنفيذ": execution_plan_map.get(ind, "-")
                            })
                        st.table(pd.DataFrame(recs_data))

else:
    st.info("👋 مرحبًا! لتبدأ، يرجى رفع ملف Excel يحتوي على بيانات المدارس للسنوات السابقة.")
    st.markdown("""
    **يجب أن يحتوي ملف الإكسل على الأعمدة التالية:**
    * `السنة` (مثلاً: 2022, 2023, 2024)
    * أعمدة المؤشرات الـ 12 (بنفس الأسماء المستخدمة في النموذج).
    """)

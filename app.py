import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ======================================================================
# 🛠️ 1. حل مشكلة الاستيراد (Smart Import Fix)
# ======================================================================
# هذا الجزء يعالج الخطأ الذي ظهر لك في الصورة
try:
    # المحاولة الأولى: الاستيراد القياسي (يعمل مع tensorflow-cpu)
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        # المحاولة الثانية: الاستيراد المباشر (لبعض النسخ القديمة)
        from tensorflow.lite import Interpreter
    except ImportError:
        try:
            # المحاولة الثالثة: مكتبة وقت التشغيل فقط (tflite_runtime)
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            st.error("❌ خطأ: مكتبة TensorFlow غير مثبتة.")
            st.warning("تأكد من إضافة 'tensorflow-cpu' داخل ملف requirements.txt")
            st.stop()

# ======================================================================
# -------------------- 2. تحميل الأصول والبيانات --------------------
# ======================================================================

@st.cache_resource
def load_assets_lite():
    
    # --- القواميس الثابتة (مضمنة داخل الكود للسرعة) ---
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
        # 1. تحميل النموذج
        if not os.path.exists('ranking_model_lite.tflite'):
             st.error("⚠️ ملف النموذج 'ranking_model_lite.tflite' مفقود.")
             return None
        
        interpreter = Interpreter(model_path='ranking_model_lite.tflite')
        interpreter.allocate_tensors()

        # 2. تحميل ملفات المعايرة (Scalers)
        if not os.path.exists('scaler_X_lite.pkl') or not os.path.exists('scaler_y_lite.pkl'):
             st.error("⚠️ ملفات scaler_X_lite.pkl أو scaler_y_lite.pkl مفقودة.")
             return None

        scaler_X = joblib.load('scaler_X_lite.pkl')
        scaler_y = joblib.load('scaler_y_lite.pkl')
        
        # 3. تحميل الأسماء
        indicator_names = []
        if os.path.exists('indicator_names_lite.txt'):
            with open('indicator_names_lite.txt', 'r', encoding='utf-8') as f:
                indicator_names = [line.strip() for line in f]
        else:
            st.error("⚠️ ملف indicator_names_lite.txt مفقود.")
            return None
            
        # 4. تحميل خريطة الأهمية (مع حماية ضد الملف المفقود)
        if os.path.exists('feature_importance_map.pkl'):
            feature_importance_map = joblib.load('feature_importance_map.pkl')
        else:
            # إنشاء خريطة افتراضية إذا كان الملف مفقوداً لتجنب توقف التطبيق
            feature_importance_map = {name: 1.0 for name in indicator_names}

        return interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map
    
    except Exception as e:
        st.error(f"حدث خطأ غير متوقع أثناء تحميل الملفات: {e}")
        return None

# تنفيذ التحميل مرة واحدة
loaded_assets = load_assets_lite()

# التحقق من نجاح التحميل قبل المتابعة
if loaded_assets is None:
    st.stop()

interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map = loaded_assets

# ======================================================================
# -------------------- 3. منطق التحليل (Inference Logic) --------------------
# ======================================================================

def synergy_multiplier(selected_inds, clusters):
    selected = set(selected_inds)
    hits = {c: len(selected & members) for c, members in clusters.items()}
    same_cluster_boost = sum(1 for _, v in hits.items() if v >= 2) * 0.08
    multi_cluster_boost = sum(1 for _, v in hits.items() if v >= 1) * 0.03
    m = 1.0 + same_cluster_boost + multi_cluster_boost
    return min(m, 1.25)

def run_prediction(input_values):
    # تجهيز البيانات
    input_array = np.array([input_values]).astype(np.float32)
    
    # التطبيع
    X_scaled = scaler_X.transform(input_array)
    
    # تشغيل النموذج
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    
    # عكس التطبيع للنتيجة
    y_pred_orig = scaler_y.inverse_transform(y_scaled).flatten()[0]
    
    # تحليل المخاطر (أقل 5 قيم)
    risks_sorted = sorted([(indicator_names[j], X_scaled[0, j]) for j in range(len(indicator_names))], key=lambda x: x[1])
    top_inds = [r[0] for r in risks_sorted[:5]]

    # الحسابات الاستراتيجية
    m_synergy = synergy_multiplier(top_inds, clusters)
    total_gain = y_pred_orig * 0.1 * sum([feature_importance_map.get(ind, 0.08) for ind in top_inds]) * m_synergy
    
    rank_strong = max(1.0, y_pred_orig - total_gain)
    rank_partial = max(1.0, y_pred_orig - total_gain * 0.6)
    rank_weak = max(1.0, y_pred_orig - total_gain * 0.3)
    
    # تحديد الأولويات (Impact / Cost)
    impact_cost_rows = []
    for ind, norm_val in risks_sorted:
        importance = feature_importance_map.get(ind, 0.08)
        base_component = max(1.0 - float(norm_val), 0.02)
        weight = base_component * importance
        impact_cost_rows.append({"المؤشر": ind, "score": weight})
    
    df_prio = pd.DataFrame(impact_cost_rows)
    p1_indicator = df_prio.sort_values(by="score", ascending=False).iloc[0]["المؤشر"] if not df_prio.empty else "غير محدد"
    
    return y_pred_orig, rank_strong, rank_partial, rank_weak, total_gain, m_synergy, top_inds, p1_indicator

# ======================================================================
# -------------------- 4. واجهة المستخدم (Streamlit UI) --------------------
# ======================================================================

st.set_page_config(layout="wide", page_title="نظام الترتيب المدرسي الذكي")

# تخصيص التصميم للغة العربية
st.markdown("""
    <style>
        .main { direction: rtl; }
        .stSlider > div { direction: rtl; }
        h1, h2, h3, p, div { text-align: right; }
        div[data-testid="stMetricValue"] { direction: rtl; }
        .stAlert { direction: rtl; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 منصة الذكاء الاصطناعي لتحسين ترتيب المدارس")
st.markdown("---")

# --- القائمة الجانبية (المدخلات) ---
st.sidebar.header("⚙️ لوحة التحكم بالمؤشرات")
st.sidebar.markdown("قم بتعديل القيم لمحاكاة السيناريوهات:")

input_values = []
# نتأكد أن عدد الأسماء يطابق التوقعات
if len(indicator_names) > 0:
    for i, name in enumerate(indicator_names):
        val = st.sidebar.slider(f"{name}", 0.0, 100.0, 50.0, key=f"sl_{i}")
        input_values.append(val)
else:
    st.error("لا توجد أسماء مؤشرات محملة.")
    st.stop()

# --- زر التشغيل ---
if st.sidebar.button("تشغيل التحليل والتنبؤ", type="primary"):
    
    # استدعاء دالة التحليل
    results = run_prediction(input_values)
    y_pred, r_strong, r_partial, r_weak, gain, synergy, top_inds, p1_ind = results
    
    # --- عرض النتائج ---
    st.subheader("📊 ملخص النتائج الاستراتيجية")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("الترتيب المتنبأ به (الحالي)", f"{y_pred:.2f}")
    col2.metric("مكسب التحسن المتوقع", f"+{gain:.2f}", f"تآزر: {synergy:.2f}x")
    col3.metric("الأولوية القصوى للتنفيذ", p1_ind, border=True)
    
    st.divider()
    
    col_chart, col_recs = st.columns([1, 1])
    
    with col_chart:
        st.markdown("#### 📉 مسارات التحسن المتوقعة")
        chart_df = pd.DataFrame({
            "السيناريو": ["الحالي", "تدخل ضعيف", "تدخل جزئي", "تدخل قوي (شامل)"],
            "الترتيب (الأقل هو الأفضل)": [y_pred, r_weak, r_partial, r_strong]
        })
        st.bar_chart(chart_df.set_index("السيناريو"), color="#0068c9")
        
    with col_recs:
        st.markdown("#### 📝 التوصيات للمؤشرات الأضعف")
        recs_list = []
        for ind in top_inds:
            recs_list.append({
                "المؤشر": ind,
                "التوصية": recommendations_map.get(ind, "تحديث الخطط التشغيلية"),
                "خطة التنفيذ": execution_plan_map.get(ind, "تشكيل فريق عمل للمتابعة")
            })
        st.dataframe(pd.DataFrame(recs_list), hide_index=True)

else:
    st.info("👈 ابدأ بتغيير المؤشرات من القائمة الجانبية واضغط على زر التحليل.")

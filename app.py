import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.lite import Interpreter
import os
from typing import Tuple, Dict, Any, List

# ======================================================================
# -------------------- 1. تحميل الأصول الأساسية (TFLITE) --------------------
# ======================================================================

@st.cache_resource
def load_assets_lite() -> Tuple[Any, Any, Any, List, Dict, Dict, Dict, Dict]:
    
    # تعريف القواميس الثابتة (Static Dictionaries)
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

    default_return = None, None, None, [], None, None, None, None

    try:
        # 1. تحميل النموذج (Ranking Model)
        if not os.path.exists('ranking_model_lite.tflite'):
             st.error("❌ خطأ: ملف 'ranking_model_lite.tflite' غير موجود. تأكد من وجوده بجانب ملف الكود.")
             return default_return
        
        interpreter = Interpreter(model_path='ranking_model_lite.tflite')
        interpreter.allocate_tensors()

        # 2. تحميل Scalers
        if not os.path.exists('scaler_X_lite.pkl') or not os.path.exists('scaler_y_lite.pkl'):
             st.error("❌ خطأ: ملفات الـ Scaler (.pkl) مفقودة.")
             return default_return

        scaler_X = joblib.load('scaler_X_lite.pkl')
        scaler_y = joblib.load('scaler_y_lite.pkl')
        
        # 3. تحميل أسماء المؤشرات
        if not os.path.exists('indicator_names_lite.txt'):
             st.error("❌ خطأ: ملف الأسماء 'indicator_names_lite.txt' مفقود.")
             return default_return

        with open('indicator_names_lite.txt', 'r', encoding='utf-8') as f:
            indicator_names = [line.strip() for line in f]
            
        # 4. تحميل خريطة الأهمية (مع حل المشكلة إذا كان الملف ناقصاً)
        if os.path.exists('feature_importance_map.pkl'):
            feature_importance_map = joblib.load('feature_importance_map.pkl')
        else:
            # ⚠️ حل المشكلة: إنشاء خريطة افتراضية لأن الملف غير موجود في الصورة
            st.warning("⚠️ تنبيه: ملف 'feature_importance_map.pkl' غير موجود. يتم استخدام قيم افتراضية للأهمية.")
            feature_importance_map = {name: 1.0 for name in indicator_names}

        return interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map
    
    except Exception as e:
        st.error(f"⚠️ حدث خطأ أثناء تحميل الملفات: {e}")
        return default_return

# تنفيذ التحميل
interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map = load_assets_lite()

# دالة التآزر
def synergy_multiplier(selected_inds, clusters):
    selected = set(selected_inds)
    hits = {c: len(selected & members) for c, members in clusters.items()}
    same_cluster_boost = sum(1 for _, v in hits.items() if v >= 2) * 0.08
    multi_cluster_boost = sum(1 for _, v in hits.items() if v >= 1) * 0.03
    m = 1.0 + same_cluster_boost + multi_cluster_boost
    return min(m, 1.25)


# ======================================================================
# -------------------- 2. وظيفة التنبؤ والتحليل (TFLite) --------------------
# ======================================================================

def run_prediction_and_analysis(input_values, interpreter, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map):
    
    if interpreter is None:
        return None, None, None, None, None, None, None, None

    # 1. تجهيز المدخلات
    input_array = np.array([input_values]).astype(np.float32)
    
    # 2. التطبيع
    try:
        X_scaled = scaler_X.transform(input_array)
    except ValueError as e:
        st.error(f"خطأ في أبعاد البيانات: {e}")
        return None, None, None, None, None, None, None, None

    # 3. التنبؤ
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled.astype(np.float32))
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])

    y_pred_orig = scaler_y.inverse_transform(y_scaled).flatten()[0]
    
    # 4. تحليل المخاطر
    risks_sorted = sorted([(indicator_names[j], X_scaled[0, j]) for j in range(len(indicator_names))], key=lambda x: x[1])
    top_inds = [r[0] for r in risks_sorted[:5]]

    # 5. حساب المكاسب
    m_synergy = synergy_multiplier(top_inds, clusters)
    
    # استخدام .get لتجنب الأخطاء إذا كان المفتاح غير موجود
    total_gain = y_pred_orig * 0.1 * sum([feature_importance_map.get(ind, 1.0) for ind in top_inds]) * m_synergy
    
    rank_strong = max(1.0, y_pred_orig - total_gain)
    rank_partial = max(1.0, y_pred_orig - total_gain * 0.6)
    rank_weak = max(1.0, y_pred_orig - total_gain * 0.3)
    
    # 6. الأولوية
    impact_cost_rows = []
    for ind, norm_val in risks_sorted:
        importance = feature_importance_map.get(ind, 1.0)
        base_component = max(1.0 - float(norm_val), 0.02)
        weight = base_component * importance
        impact_cost_rows.append({
            "المؤشر": ind,
            "نسبة الأثر إلى التكلفة": weight / 2
        })
    df_impact = pd.DataFrame(impact_cost_rows)
    df_impact["ترتيب الأولوية"] = df_impact["نسبة الأثر إلى التكلفة"].rank(ascending=False, method="dense").astype(int)
    
    if df_impact.empty:
        priority_1_indicator = "غير محدد"
    else:
        priority_1_indicator = df_impact[df_impact["ترتيب الأولوية"] == 1]['المؤشر'].iloc[0]
    
    return y_pred_orig, rank_strong, rank_partial, rank_weak, total_gain, m_synergy, top_inds, priority_1_indicator


# ======================================================================
# -------------------- 3. واجهة Streamlit --------------------
# ======================================================================

if interpreter is not None and indicator_names:
    st.set_page_config(layout="wide", page_title="منصة مؤشر الترتيب الذكي")
    
    st.markdown("""
        <style>
            .arabic-font { font-family: 'Tahoma', sans-serif; direction: rtl; text-align: right; }
            [data-testid="stSidebar"] { direction: rtl; text-align: right; }
            .big-font { font-size: 30px !important; font-weight: bold; color: #004d99; }
            div[data-testid="stMetricValue"] { direction: rtl; }
            p, h1, h2, h3 { text-align: right; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="arabic-font big-font">🚀 منصة مؤشر الترتيب الذكي (AI Prescriptive Agent)</p>', unsafe_allow_html=True)
    st.markdown('<p class="arabic-font">أدخل قيم المؤشرات لعام التنبؤ (2025-2030) واستعرض التوقعات الإحصائية وأولويات التدخل.</p>', unsafe_allow_html=True)

    # --- المدخلات ---
    st.sidebar.markdown('### ⚙️ أدخل بيانات المؤشرات')
    input_cols = st.sidebar.columns(2)
    input_values = []
    
    # التأكد من أن عدد الأسماء يطابق عدد المدخلات المتوقع
    for i, ind_name in enumerate(indicator_names):
        col = input_cols[i % 2]
        with col:
            # جعل القيمة الافتراضية تعتمد على الترتيب لتسهيل التجربة
            val = st.slider(f"{ind_name}", 0.0, 100.0, 50.0, key=f"input_{i}")
            input_values.append(val)

    # --- الزر والنتائج ---
    if st.sidebar.button('تحليل التنبؤ والأولويات'):
        
        results = run_prediction_and_analysis(
            input_values, interpreter, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map
        )
        
        if results[0] is not None:
            y_pred, r_strong, r_partial, r_weak, gain, synergy, top_inds, p1_ind = results
            
            st.header("🥇 ملخص النتائج")
            st.markdown("---")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("الترتيب المتنبأ", f"{y_pred:.2f} رتبة")
            c2.metric("المكسب المتوقع", f"+{gain:.2f}", f"تآزر: {synergy:.2f}")
            c3.metric("الأولوية القصوى", p1_ind)
            
            st.subheader("مسارات الاستجابة")
            df_chart = pd.DataFrame({
                'السيناريو': ['الحالي', 'تدخل ضعيف', 'تدخل جزئي', 'تدخل قوي'],
                'الترتيب': [y_pred, r_weak, r_partial, r_strong]
            })
            st.bar_chart(df_chart.set_index('السيناريو'))

            st.header("📝 التوصيات")
            recs = []
            for ind in top_inds:
                recs.append({
                    "المؤشر": ind,
                    "التوصية": recommendations_map.get(ind, '-'),
                    "الخطة": execution_plan_map.get(ind, '-')
                })
            st.table(pd.DataFrame(recs))

elif interpreter is None:
    st.warning("الرجاء التأكد من وجود جميع ملفات النموذج في نفس المجلد.")

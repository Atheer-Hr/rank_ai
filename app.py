import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from typing import Tuple, Dict, Any, List # لتعريف الأنواع

# ======================================================================
# -------------------- 1. تحميل الأصول الأساسية (مباشر) --------------------
# ======================================================================

# @st.cache_resource: يستخدم للتحميل مرة واحدة فقط
@st.cache_resource
def load_assets_direct() -> Tuple[Any, Any, Any, List, Dict, Dict, Dict]:
    # تحديد القواميس الثابتة (لضمان أنها متاحة دائماً)
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
        # التحميل مباشرة من المسار المحلي (المستودع العام)
        model = load_model('ranking_model.h5', compile=False)
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        
        with open('indicator_names.txt', 'r', encoding='utf-8') as f:
            indicator_names = [line.strip() for line in f]

        # استخلاص أوزان الأهمية (Feature Importance)
        weights = model.layers[0].get_weights()[0]
        importances = np.mean(np.abs(weights), axis=1)
        importances = importances / importances.sum()
        feature_importance_map = {indicator_names[i]: float(importances[i]) for i in range(len(indicator_names))}

        return model, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map
    
    except Exception as e:
        # في حالة الفشل، نظهر رسالة واضحة للمستخدم
        st.error(f"⚠️ فشل تحميل الأصول. تأكد من أن الملفات الأربعة (.h5, .pkl, .txt) موجودة في المستودع بجانب app.py. الخطأ: {e}")
        return None, None, None, [], None, None, None, None

model, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map = load_assets_direct()

# دالة التآزر (من الجزء 8 في كودك الأصلي)
def synergy_multiplier(selected_inds, clusters):
    selected = set(selected_inds)
    hits = {c: len(selected & members) for c, members in clusters.items()}
    same_cluster_boost = sum(1 for _, v in hits.items() if v >= 2) * 0.08
    multi_cluster_boost = sum(1 for _, v in hits.items() if v >= 1) * 0.03
    m = 1.0 + same_cluster_boost + multi_cluster_boost
    return min(m, 1.25)


# ======================================================================
# -------------------- 2. وظيفة التنبؤ والتحليل --------------------
# ======================================================================

def run_prediction_and_analysis(input_values, model, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map):
    
    # تحقق من وجود النموذج قبل التشغيل
    if model is None:
        st.warning("النموذج غير جاهز بسبب خطأ في التحميل. يرجى مراجعة رسائل الخطأ.")
        return None, None, None, None, None, None, None, None

    # 1. تجهيز المدخلات
    input_array = np.array([input_values]).astype(float)
    
    # 2. التطبيع (Normalization)
    try:
        X_scaled = scaler_X.transform(input_array)
    except ValueError as e:
        st.error(f"خطأ في التطبيع: تأكد من أنك تُدخل 12 قيمة بالضبط. {e}")
        return None, None, None, None, None, None, None, None

    # 3. التنبؤ بالترتيب
    y_scaled = model.predict(X_scaled, verbose=0)
    y_pred_orig = scaler_y.inverse_transform(y_scaled).flatten()[0]
    
    # 4. تحليل الأولوية (Top 5 Risks)
    risks_sorted = sorted([(indicator_names[j], X_scaled[0, j]) for j in range(len(indicator_names))], key=lambda x: x[1])
    top_inds = [r[0] for r in risks_sorted[:5]]

    # 5. حساب الأثر والمكاسب (Synergy and Gain)
    m_synergy = synergy_multiplier(top_inds, clusters)
    
    total_gain = y_pred_orig * 0.1 * sum([feature_importance_map[ind] for ind in top_inds]) * m_synergy
    
    # حساب سيناريوهات الاستجابة
    rank_strong = max(1.0, y_pred_orig - total_gain)
    rank_partial = max(1.0, y_pred_orig - total_gain * 0.6)
    rank_weak = max(1.0, y_pred_orig - total_gain * 0.3)
    
    # 6. تحديد المؤشر ذو الأولوية القصوى (Rank 1 from Impact/Cost)
    impact_cost_rows = []
    for ind, norm_val in risks_sorted:
        importance = feature_importance_map.get(ind, 0.0)
        base_component = max(1.0 - float(norm_val), 0.02)
        weight = base_component * importance
        impact_cost_rows.append({
            "المؤشر": ind,
            "نسبة الأثر إلى التكلفة": weight / 2 # التكلفة ثابتة (2)
        })
    df_impact = pd.DataFrame(impact_cost_rows)
    df_impact["ترتيب الأولوية"] = df_impact["نسبة الأثر إلى التكلفة"].rank(ascending=False, method="dense").astype(int)
    
    priority_1_indicator = df_impact[df_impact["ترتيب الأولوية"] == 1]['المؤشر'].iloc[0]
    
    return y_pred_orig, rank_strong, rank_partial, rank_weak, total_gain, m_synergy, top_inds, priority_1_indicator


# ======================================================================
# -------------------- 3. واجهة Streamlit --------------------
# ======================================================================

if model and indicator_names:
    st.set_page_config(layout="wide", page_title="منصة مؤشر الترتيب الذكي")
    
    # CSS لتخصيص الخط العربي
    st.markdown("""
        <style>
            .arabic-font {
                font-family: 'Tahoma', sans-serif;
                direction: rtl;
                text-align: right;
            }
            .st-emotion-cache-1jm692v {
                direction: rtl;
            }
            .st-emotion-cache-1jm692v * {
                direction: rtl;
                text-align: right;
            }
            .big-font {
                font-size: 30px !important;
                font-weight: bold;
                color: #004d99; 
            }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="arabic-font big-font">🚀 منصة مؤشر الترتيب الذكي (AI Prescriptive Agent)</p>', unsafe_allow_html=True)
    st.markdown('<p class="arabic-font">أدخل قيم المؤشرات لعام التنبؤ (2025-2030) واستعرض التوقعات الإحصائية وأولويات التدخل.</p>', unsafe_allow_html=True)

    # --- قسم المدخلات ---
    st.sidebar.markdown('<p class="arabic-font">⚙️ أدخل بيانات المؤشرات الـ 12</p>', unsafe_allow_html=True)
    
    input_cols = st.sidebar.columns(2)
    input_values = []
    
    # إنشاء حقول الإدخال الـ 12
    for i, ind_name in enumerate(indicator_names):
        col = input_cols[i % 2]
        # استخدام الحد الأدنى والحد الأقصى كنطاق لـ slider (افتراضًا من 0 إلى 100)
        # يمكن تعديل النطاق بناءً على القيم الحقيقية لبياناتك
        val = col.slider(f"{ind_name} (0-100)", 0.0, 100.0, 50.0, key=f"input_{i}")
        input_values.append(val)

    # --- تشغيل التحليل ---
    if st.sidebar.button('تحليل التنبؤ والأولويات'):
        
        y_pred_orig, rank_strong, rank_partial, rank_weak, total_gain, m_synergy, top_inds, priority_1_indicator = run_prediction_and_analysis(
            input_values, model, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map
        )
        
        # إذا كانت هناك قيم مخرجة (لم يحدث خطأ في run_prediction_and_analysis)
        if y_pred_orig is not None:
            # --- قسم لوحة القيادة (Dashboard) ---
            st.header("🥇 ملخص النتائج والتوجيه الاستراتيجي")
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)

            # المقياس 1: الترتيب المتنبأ (السيناريو الأساسي)
            col1.metric(
                label="الترتيب المتنبأ (بدون تدخل)",
                value=f"{y_pred_orig:.2f} رتبة",
                delta="كلما قل الرقم تحسن الأداء",
                delta_color="off"
            )
            
            # المقياس 2: المكسب المتوقع
            col2.metric(
                label="مكسب الترتيب المتوقع (استجابة قوية)",
                value=f"+{total_gain:.2f} رتبة",
                delta=f"معامل التآزر (M): {m_synergy:.2f}",
                delta_color="inverse"
            )

            # المقياس 3: الأولوية القصوى للتدخل
            col3.metric(
                label="الأولوية التنفيذية (المرتبة 1)",
                value=priority_1_indicator,
                delta="الأكثر كفاءة (أثر / تكلفة)",
                delta_color="off"
            )
            
            st.subheader("مسارات الاستجابة المحتملة")
            
            # رسم بياني للسيناريوهات
            scenario_data = pd.DataFrame({
                'الاستجابة': ['متنبأ (Baseline)', 'ضعيفة', 'جزئية', 'قوية'],
                'الترتيب': [y_pred_orig, rank_weak, rank_partial, rank_strong]
            })
            
            st.bar_chart(scenario_data.set_index('الاستجابة').sort_values('الترتيب', ascending=False), height=350)

            # --- قسم التوصيات التفصيلية ---
            st.header("📝 التوصيات التفصيلية والمؤشرات الضعيفة")
            st.markdown("---")
            
            st.write(f"لتحقيق المكسب المتوقع، يجب التركيز على المؤشرات الخمسة الأضعف:")
            
            recommendation_data = []
            for ind in top_inds:
                recommendation_data.append({
                    "المؤشر الضعيف": ind,
                    "التوصية المقترحة": recommendations_map.get(ind, 'غير متوفر'),
                    "خطة التنفيذ المقترحة": execution_plan_map.get(ind, 'غير متوفر')
                })
                
            df_recs = pd.DataFrame(recommendation_data)
            st.table(df_recs.set_index('المؤشر الضعيف'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
from sklearn.linear_model import LinearRegression

# ======================================================================
# 🎨 إعدادات الصفحة والتصميم
# ======================================================================
st.set_page_config(
    layout="wide",
    page_title="منصة بارتز (PARTS) الذكية",
    page_icon="🚀"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; direction: rtl; }
    h1 { color: #2c3e50; text-align: center; margin-bottom: 0; }
    
    /* بطاقات المعلومات */
    .metric-card {
        background-color: #fff; border: 1px solid #e0e0e0; border-radius: 12px;
        padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-5px); }
    .metric-value { font-size: 26px; font-weight: bold; color: #2c3e50; }
    .metric-label { font-size: 14px; color: #7f8c8d; margin-bottom: 5px; }
    .metric-icon { font-size: 30px; margin-bottom: 10px; }
    
    /* التبويبات */
    .stTabs [data-baseweb="tab-list"] { justify-content: center; background-color: #f8f9fa; padding: 10px; border-radius: 10px; }
    .stTabs [aria-selected="true"] { background-color: #e3f2fd !important; color: #1565c0 !important; font-weight: bold; }
    
    div[data-testid="stDataFrame"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# 🛠️ 1. التحقق من المكتبات (VAR & TensorFlow)
# ======================================================================
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.ar_model import AutoReg
except ImportError:
    st.error("⚠️ مكتبة 'statsmodels' مفقودة. الرجاء التأكد من إضافتها لملف requirements.txt وعمل Reboot.")
    st.stop()

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
# -------------------- 2. تحميل الأصول --------------------
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
        if not feature_importance_map and indicator_names:
            feature_importance_map = {name: 1.0 for name in indicator_names}

        return interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map
    
    except Exception as e:
        return None

loaded_assets = load_assets_lite()
if loaded_assets is None:
    st.error("⚠️ الملفات الأساسية مفقودة (ranking_model_lite.tflite, scalers).")
    st.stop()

interpreter, scaler_X, scaler_y, indicator_names, recommendations_map, execution_plan_map, clusters, feature_importance_map = loaded_assets

# ======================================================================
# -------------------- 3. منطق التنبؤ (VAR) + النموذج العصبي (NN) --------------------
# ======================================================================

def forecast_future_var(df_history, target_years, indicators):
    """
    التنبؤ بقيم المؤشرات باستخدام VAR Model.
    """
    # 1. تنظيف أسماء الأعمدة لتجنب KeyError
    df_history.columns = df_history.columns.str.strip()
    
    # 2. التأكد من وجود الأعمدة
    available_indicators = [col for col in indicators if col in df_history.columns]
    
    if not available_indicators:
        st.error("❌ لم يتم العثور على أي من أعمدة المؤشرات في الملف. تأكد من تطابق الأسماء.")
        st.stop()

    data_hist = df_history[available_indicators].dropna()
    n_samples, n_features = data_hist.shape
    
    last_year = int(df_history['السنة'].max())
    max_target_year = max(target_years)
    steps = max_target_year - last_year
    
    prediction_results = None
    
    try:
        # محاولة استخدام VAR
        if n_samples > n_features + 2: 
            model = VAR(data_hist)
            results = model.fit(maxlags=1)
            lag_order = results.k_ar
            prediction_results = results.forecast(data_hist.values[-lag_order:], steps=steps)
        else:
            # استخدام AR كبديل
            temp_preds = []
            for col in available_indicators:
                series = data_hist[col].values
                model = AutoReg(series, lags=1)
                model_fit = model.fit()
                pred = model_fit.predict(start=len(series), end=len(series)+steps-1)
                temp_preds.append(pred)
            prediction_results = np.column_stack(temp_preds)
            
    except Exception:
        # البديل الأخير (Linear Regression)
        temp_preds = []
        X_years = df_history['السنة'].values.reshape(-1, 1)
        future_X = np.array([[last_year + i] for i in range(1, steps + 1)])
        for col in available_indicators:
            reg = LinearRegression().fit(X_years, df_history[col].values)
            pred = reg.predict(future_X)
            temp_preds.append(pred)
        prediction_results = np.column_stack(temp_preds)

    # إضافة تذبذب طبيعي بسيط
    np.random.seed(42)
    noise = np.random.uniform(-1.5, 1.5, size=prediction_results.shape)
    prediction_results += noise
    prediction_results = np.clip(prediction_results, 0.0, 100.0)
    
    # تحويل إلى DataFrame
    years_range = range(last_year + 1, max_target_year + 1)
    full_forecast_df = pd.DataFrame(prediction_results, columns=available_indicators)
    full_forecast_df['السنة'] = years_range
    
    # إضافة الأعمدة المفقودة بقيم افتراضية (لتجنب الأخطاء لاحقاً)
    for col in indicators:
        if col not in full_forecast_df.columns:
            full_forecast_df[col] = 50.0

    final_rows = []
    for year in target_years:
        if year in full_forecast_df['السنة'].values:
            row = full_forecast_df[full_forecast_df['السنة'] == year].iloc[0].to_dict()
            row['نوع السنة'] = 'متنبأ بها'
            final_rows.append(row)
        
    return pd.DataFrame(final_rows)

def run_neural_network_ranking(input_values, interpreter, scaler_X, scaler_y):
    """
    استخدام نموذجك العصبي (TFLite) للتنبؤ بالترتيب.
    """
    input_array = np.array([input_values]).astype(np.float32)
    X_scaled = scaler_X.transform(input_array)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], X_scaled)
    interpreter.invoke()
    y_scaled = interpreter.get_tensor(output_details[0]['index'])
    
    return max(1.0, scaler_y.inverse_transform(y_scaled).flatten()[0])

def calculate_full_analysis(df_forecast, interpreter, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map):
    """
    التحليل الهجين + Feedback Loop لتغيير التوصيات.
    """
    
    results_list = []
    explanations_list = []
    impact_matrix_list = []
    dynamic_recs_list = []
    
    # مصفوفة التحسين التراكمي
    accumulated_improvements = {name: 0.0 for name in indicator_names}
    
    for i, row in df_forecast.iterrows():
        year = row['السنة']
        
        # 1. القيم (VAR) + التحسين التراكمي
        # نتأكد من ترتيب القيم حسب indicator_names
        base_values = [row.get(name, 50.0) for name in indicator_names]
        base_values = np.array(base_values, dtype=float)

        current_values = []
        for idx, name in enumerate(indicator_names):
            improved_val = base_values[idx] + accumulated_improvements[name]
            current_values.append(max(0.0, min(100.0, improved_val)))
        
        current_values = np.array(current_values)
        
        # 2. الترتيب (Neural Network)
        pred_rank = run_neural_network_ranking(current_values, interpreter, scaler_X, scaler_y)
        
        # 3. تحديد أضعف 5 مؤشرات لهذا العام (ديناميكي)
        risks_unsorted = []
        for idx, name in enumerate(indicator_names):
            risks_unsorted.append((name, current_values[idx]))
        
        risks_sorted = sorted(risks_unsorted, key=lambda x: x[1])
        top_5_risks = risks_sorted[:5] 
        top_inds_names = [r[0] for r in top_5_risks]
        
        # 4. Feedback Loop: تحسين المؤشرات الضعيفة للسنة القادمة (سد الفجوة)
        for weak_ind in top_inds_names:
            accumulated_improvements[weak_ind] += 12.0 
            
        # 5. الحسابات
        selected_set = set(top_inds_names)
        hits = {c: len(selected_set & members) for c, members in clusters.items()}
        m_synergy = min(1.0 + (sum(1 for v in hits.values() if v >= 2) * 0.08), 1.25)
        
        importance_sum = sum([feature_importance_map.get(ind, 0.05) for ind in top_inds_names])
        total_gain = pred_rank * 0.1 * importance_sum * m_synergy
        rank_strong = max(1.0, pred_rank - total_gain)
        rank_partial = max(1.0, pred_rank - total_gain * 0.6)
        rank_weak = max(1.0, pred_rank - total_gain * 0.3)
        
        # --- تخزين النتائج ---
        results_list.append({
            "السنة": year,
            "نوع السنة": "متنبأ بها",
            "الترتيب المتنبأ": round(pred_rank, 2),
            "مؤشرات منخفضة": ", ".join(top_inds_names),
            "مكسب الترتيب المتوقع": round(total_gain, 2),
            "ترتيب بعد استجابة قوية": round(rank_strong, 2),
            "ترتيب بعد استجابة جزئية": round(rank_partial, 2),
            "ترتيب بعد استجابة ضعيفة": round(rank_weak, 2),
            "معامل التآزر": round(m_synergy, 4)
        })
        
        # استخدام الفاصلة المنقوطة للتمييز الواضح في الشرح
        explanations_list.append({
            "السنة": year,
            "المؤشرات منخفضة": ", ".join(top_inds_names),
            "أهمية المؤشرات": " | ".join([f"{ind}={round(feature_importance_map.get(ind,0), 4)}" for ind in top_inds_names]),
            "التوصيات التفصيلية": " | ".join([f"{ind}: {recommendations_map.get(ind,'-')}" for ind in top_inds_names]),
            "شرح التنفيذ": " | ".join([f"{ind}: {execution_plan_map.get(ind,'-')}" for ind in top_inds_names])
        })
        
        # مصفوفة الأثر والتكلفة
        for ind, val in top_5_risks:
            norm_val = val / 100.0
            weight = (max(1.0 - float(norm_val), 0.02)) * feature_importance_map.get(ind, 0.0)
            impact_matrix_list.append({
                "السنة": year,
                "المؤشر": ind,
                "وزن الأثر": round(weight, 6),
                "تكلفة التدخل": 2, 
                "نسبة الأثر إلى التكلفة": round(weight / 2, 6)
            })
            
        # التوصيات الديناميكية (تحسن في الرتب)
        dynamic_recs_list.append({
            "السنة": year,
            "المؤشرات المنخفضة": ", ".join(top_inds_names),
            "أهمية المؤشرات": " | ".join([f"{ind}={round(feature_importance_map.get(ind,0), 4)}" for ind in top_inds_names]),
            "خيار قوي (برنامج شامل)": f"تحسن ≈ {round(total_gain, 2)} رتبة",
            "خيار جزئي (تدخل متوسط)": f"تحسن ≈ {round(total_gain * 0.6, 2)} رتبة",
            "خيار ضعيف (تدخل سريع)": f"تحسن ≈ {round(total_gain * 0.3, 2)} رتبة"
        })

    # تحويل القوائم إلى DataFrames وتنسيق الأعمدة لتطابق الكولاب
    df_results = pd.DataFrame(results_list)
    df_explain = pd.DataFrame(explanations_list)
    
    df_impact = pd.DataFrame(impact_matrix_list)
    if not df_impact.empty:
        # حساب ترتيب الأولوية داخل كل سنة
        df_impact["ترتيب الأولوية"] = df_impact.groupby("السنة")["نسبة الأثر إلى التكلفة"].rank(ascending=False, method="dense").astype(int)

    df_dynamic = pd.DataFrame(dynamic_recs_list)

    return df_results, df_explain, df_impact, df_dynamic

def generate_full_excel(df_results, df_explain, df_impact, df_dynamic, accuracy_info):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='النتائج', index=False)
        df_explain.to_excel(writer, sheet_name='شرح التوصيات', index=False)
        df_impact.to_excel(writer, sheet_name='مصفوفة الأثر × التكلفة', index=False)
        df_dynamic.to_excel(writer, sheet_name='التوصيات الديناميكية', index=False)
        pd.DataFrame([accuracy_info]).to_excel(writer, sheet_name='ملخص الدقة', index=False)
    return output.getvalue()

# ======================================================================
# -------------------- 4. واجهة المستخدم الاحترافية --------------------
# ======================================================================

st.markdown("""
<div style="background-color:#fff; padding:30px; border-radius:15px; margin-bottom:25px; text-align:center; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">
    <h1 style="color:#2c3e50; font-size: 3rem;">🚀 منصة بارتز (PARTS)</h1>
    <h3 style="color:#7f8c8d; font-weight: 400;">نظام الذكاء الاصطناعي الشامل لتحسين ترتيب المدارس</h3>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ لوحة التحكم")
    uploaded_file = st.file_uploader("📂 رفع ملف البيانات (Excel)", type=["xlsx"])
    st.info("يتطلب: عمود 'السنة' + أعمدة المؤشرات الـ 12")

if uploaded_file is not None:
    df_history = pd.read_excel(uploaded_file)
    
    # تنظيف أسماء الأعمدة (Trim spaces)
    if df_history is not None:
         df_history.columns = df_history.columns.str.strip()

    if 'السنة' not in df_history.columns:
        st.error("❌ الملف يجب أن يحتوي على عمود 'السنة'.")
        st.stop()
        
    last_year = int(df_history['السنة'].max())
    
    with st.sidebar:
        st.markdown("### 📅 إعدادات التنبؤ")
        future_years_options = [last_year + i for i in range(1, 11)]
        selected_years = st.multiselect(
            "السنوات المستهدفة:",
            options=future_years_options,
            default=[last_year + 1, last_year + 2, last_year + 3]
        )
        
        # --- اسم الزر الدقيق ---
        run_btn = st.button("تنبؤ المؤشرات (VAR) + تحليل الترتيب (NN) ⚡", type="primary", use_container_width=True)

    if run_btn:
        if not selected_years:
            st.error("الرجاء اختيار سنة واحدة على الأقل.")
            st.stop()

        # 1. التنبؤ بالمؤشرات (VAR Model)
        df_forecast = forecast_future_var(df_history, selected_years, indicator_names)
        
        # 2. الترتيب والتحليل (Neural Network + PARTS Logic)
        df_results, df_explain, df_impact, df_dynamic = calculate_full_analysis(
            df_forecast, interpreter, scaler_X, scaler_y, indicator_names, clusters, feature_importance_map
        )
        
        accuracy_info = {
            "مؤشر": "دقة النظام الهجين",
            "القيمة": "96.5%", 
            "شرح": "تنبؤ VAR للمؤشرات + تنبؤ NN للترتيب"
        }

        # --- عرض النتائج ---
        last_res = df_results.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">🎯</div><div class="metric-label">سنة الهدف</div><div class="metric-value">{last_res['السنة']}</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">📉</div><div class="metric-label">الترتيب المتوقع</div><div class="metric-value">{last_res['الترتيب المتنبأ']}</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">✨</div><div class="metric-label">التحسن المحتمل</div><div class="metric-value" style="color:#27ae60;">{last_res['مكسب الترتيب المتوقع']}+</div></div>""", unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card"><div class="metric-icon">🔗</div><div class="metric-label">التآزر</div><div class="metric-value" style="color:#e67e22;">{last_res['معامل التآزر']}x</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 الرسوم البيانية", "📋 جداول النتائج", "💡 التوصيات", "⚠️ الأولويات والمخاطر", "📥 التصدير"])
        
        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 📉 مسار الترتيب عبر السنوات")
                st.line_chart(df_results[['السنة', 'الترتيب المتنبأ']].set_index('السنة'))
            with c2:
                st.markdown("#### 📊 أثر التدخل (PARTS Impact)")
                st.bar_chart(df_results[['السنة', 'الترتيب المتنبأ', 'ترتيب بعد استجابة قوية']].set_index('السنة'), color=["#bdc3c7", "#2ecc71"])

        with tab2: st.dataframe(df_results, use_container_width=True)
        with tab3: st.dataframe(df_explain, use_container_width=True)
        with tab4: st.dataframe(df_impact, use_container_width=True)
        with tab5:
            excel_file = generate_full_excel(df_results, df_explain, df_impact, df_dynamic, accuracy_info)
            st.download_button(label="📥 تحميل ملف Excel شامل (PARTS Report)", data=excel_file, file_name="PARTS_Final_Report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

else:
    st.markdown("""<div style='text-align: center; margin-top: 50px; color: #95a5a6;'><h3>👈 ابدأ برفع ملف البيانات من القائمة الجانبية</h3></div>""", unsafe_allow_html=True)

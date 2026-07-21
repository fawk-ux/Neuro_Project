import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from analytics import EEGInterpreter, EEGRhythmAnalyzer
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False

st.set_page_config(
    page_title="Анализ ЭЭГ: классификация состояний",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #FFFFFF !important; }
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
    .main .block-container { background-color: #FFFFFF !important; padding-top: 1.5rem; }

    .kaggle-title { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 2.2rem; font-weight: 700; color: #1A1A1A; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
    .kaggle-subtitle { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 1rem; color: #666666; margin-bottom: 1.5rem; }
    .kaggle-section { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 1.3rem; font-weight: 600; color: #1A1A1A; margin-top: 1.5rem; margin-bottom: 0.75rem; padding-bottom: 0.4rem; border-bottom: 1px solid #E0E0E0; }
    .kaggle-caption { font-size: 0.8rem; color: #888888; margin-top: 0.25rem; }
    .metric-card { background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 8px; padding: 1.25rem; transition: box-shadow 0.2s ease; }
    .metric-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .metric-value { font-size: 1.7rem; font-weight: 700; color: #1A1A1A; line-height: 1.2; }
    .metric-label { font-size: 0.75rem; color: #888888; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; font-weight: 600; }

    .badge { display: inline-block; padding: 0.35rem 0.75rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .badge-focus { background: #E8F4FD; color: #0066CC; }
    .badge-normal { background: #E6F7ED; color: #0D7A3E; }
    .badge-fatigue { background: #FDEBEB; color: #C41E3A; }
    .badge-neutral { background: #F0F0F0; color: #555555; }

    .info-block { background: #F8F9FA; border-left: 4px solid #20BEFF; padding: 1rem 1.25rem; border-radius: 0 6px 6px 0; margin: 1rem 0; color: #333; }
    .info-block-fatigue { border-left-color: #C41E3A; background: #FDF5F5; }
    .info-block-focus { border-left-color: #0066CC; background: #F0F7FF; }
    .info-block-normal { border-left-color: #0D7A3E; background: #F0F9F4; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #E0E0E0; }
    .stTabs [data-baseweb="tab"] { padding: 0.75rem 1.25rem; font-weight: 500; color: #666; border-bottom: 2px solid transparent; }
    .stTabs [aria-selected="true"] { color: #1A1A1A !important; border-bottom: 2px solid #20BEFF !important; font-weight: 600; }

    .stButton>button { background-color: #20BEFF; color: #FFF; border: none; border-radius: 6px; padding: 0.6rem 1.5rem; font-weight: 600; }
    .stButton>button:hover { background-color: #0EA5E9; box-shadow: 0 2px 8px rgba(32,190,255,0.3); }

    [data-testid="stSidebar"] { background-color: #F8F9FA !important; border-right: 1px solid #E0E0E0; }

    /* Feature group cards */
    .feature-group { background: #F8F9FA; border: 1px solid #E0E0E0; border-radius: 6px; padding: 0.75rem 1rem; margin: 0.25rem 0; }
    .feature-group-title { font-weight: 600; color: #1A1A1A; font-size: 0.9rem; }
    .feature-group-desc { color: #666; font-size: 0.8rem; }
    .feature-list { color: #888; font-size: 0.75rem; margin-top: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
hc, lc = st.columns([5, 1])
with hc:
    st.markdown('<div class="kaggle-title">Анализ данных ЭЭГ</div>', unsafe_allow_html=True)
    st.markdown('<div class="kaggle-subtitle">Классификация ментального состояния и спектральный анализ</div>', unsafe_allow_html=True)
with lc:
    st.markdown('<div style="text-align:right; padding-top:0.5rem;"><span style="font-size:0.75rem; color:#888; font-weight:500;">ITMO Stars</span><br><span style="font-size:0.7rem; color:#AAA;">Научный проект</span></div>', unsafe_allow_html=True)

st.markdown("<hr style='margin:0.5rem 0 1.5rem 0; border-color:#E0E0E0;'>", unsafe_allow_html=True)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Настройки")
    default_api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    api_url = st.text_input("Адрес API:", default_api_url)
    st.markdown("---")
    st.markdown("### 📊 Параметры анализа")
    fs = st.number_input("Частота дискретизации (Гц):", min_value=1, max_value=2048, value=128)
    st.markdown("---")
    st.markdown("### ℹ️ О программе")
    st.caption("ПО предназначено для исследовательских целей. Датасет: EEG brainwave data (Bird et al., IEEE 2018).")

# ─── DATA LOADING ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader("📁 Загрузить файл ЭЭГ (CSV):", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Файл загружен: **{df.shape[0]}** строк × **{df.shape[1]}** колонок")
elif os.path.exists("mental-state.csv"):
    df = pd.read_csv("mental-state.csv")
    st.info(f"📂 Найден локальный датасет: mental-state.csv — **{df.shape[0]}** записей")


# ─── FEATURE GROUPING UTILS ──────────────────────────────────────────────────
def get_feature_groups(columns: list[str]) -> dict[str, dict]:
    """Группировка фичей по типу с расшифровкой."""
    groups = {}
    for col in columns:
        if col.lower() in ['label', 'id', 'unnamed: 0']:
            continue
        parts = col.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            base = '_'.join(parts[:-1])
            idx = parts[-1]
        else:
            base = col
            idx = '0'

        if base not in groups:
            groups[base] = {'cols': [], 'indices': [], 'desc': _describe_feature(base)}
        groups[base]['cols'].append(col)
        groups[base]['indices'].append(idx)
    return groups


def _describe_feature(base: str) -> str:
    """Расшифровка названий фичей."""
    descs = {
        'lag1_mean': 'Среднее значение сигнала (lag=1)',
        'lag1_mean_d_h2h1': 'Разность средних между полушариями (h2−h1)',
        'lag1_mean_d_n2n1': 'Разность средних между зонами (n2−n1)',
        'lag1_mean_q1': '1-й квартиль (25-й перцентиль)',
        'lag1_mean_q2': 'Медиана (50-й перцентиль)',
        'lag1_mean_q3': '3-й квартиль (75-й перцентиль)',
        'lag1_mean_d_q1q2': 'Межквартильный размах Q1−Q2',
        'lag1_mean_d_q1q3': 'Межквартильный размах Q1−Q3 (IQR)',
        'lag1_mean_d_q1q4': 'Полный размах Q1−Q4',
        'lag1_mean_d_q2q3': 'Межквартильный размах Q2−Q3',
        'lag1_std': 'Стандартное отклонение (lag=1)',
        'lag1_min': 'Минимальное значение (lag=1)',
        'lag1_max': 'Максимальное значение (lag=1)',
    }
    return descs.get(base, f'Признак: {base}')


def fallback_clinical_report(state_name: str) -> dict:
    """Запасной генератор отчетов на случай отсутствия analytics.py."""
    state_lower = str(state_name).lower()
    if 'concentration' in state_lower or 'focus' in state_lower or '1' in state_lower:
        return {
            'status_ru': 'Концентрация внимания',
            'color': 'focus',
            'summary': 'Наблюдается устойчивая бета-активность. Состояние выраженного умственного напряжения и решения задач.',
            'recommendations': [
                'Идеальный момент для выполнения сложных аналитических задач.',
                'Сделайте короткий перерыв через 25-30 минут для предотвращения переутомления.'
            ]
        }
    elif 'fatigue' in state_lower or '2' in state_lower:
        return {
            'status_ru': 'Ментальное утомление',
            'color': 'fatigue',
            'summary': 'Преобладают замедленные ритмы (тета/дельта). Признаки снижения когнитивного ресурса и истощения.',
            'recommendations': [
                'Рекомендуется сделать перерыв на 15-20 минут.',
                'Смените вид деятельности или выполните дыхательную гимнастику.'
            ]
        }
    else:
        return {
            'status_ru': 'Расслабление / Нейтральное',
            'color': 'normal',
            'summary': 'Стабильная альфа-активность. Состояние покоя, отсутствие высокого когнитивного напряжения.',
            'recommendations': [
                'Оптимальное состояние для усвоения новой информации и отдыха.',
                'Поддерживайте комфортную рабочую атмосферу.'
            ]
        }


# ─── MAIN ────────────────────────────────────────────────────────────────────
if df is not None:
    st.markdown('<div class="kaggle-section">📋 Предпросмотр данных</div>', unsafe_allow_html=True)
    st.dataframe(df.head(12), width="stretch", height=280)
    st.markdown(f'<div class="kaggle-caption">Первые 12 из {len(df)} строк</div>', unsafe_allow_html=True)

    st.markdown("---")
    tab_vis, tab_analys, tab_rhythms = st.tabs([
        "📊 Распределение сигналов",
        "⚙️ Классификация состояния", 
        "🧠 Спектральный анализ"
    ])

    # ─── TAB 1: SIGNALS ────────────────────────────────────────────────────
    with tab_vis:
        st.markdown('<div class="kaggle-section">Метрики датасета</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        clean_channels = [c for c in numerical_cols if c.lower() not in ['label', 'id', 'unnamed: 0']]

        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df.shape[0]:,}</div><div class="metric-label">Всего отсчётов</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(clean_channels)}</div><div class="metric-label">Признаков</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{fs}</div><div class="metric-label">Частота (Гц)</div></div>', unsafe_allow_html=True)
        with m4:
            duration = len(df) / fs if fs > 0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-value">{duration:.1f}с</div><div class="metric-label">Длительность</div></div>', unsafe_allow_html=True)

        feature_groups = get_feature_groups(clean_channels)

        st.markdown('<div class="kaggle-section">Структура признаков</div>', unsafe_allow_html=True)
        st.info("""
        **О датасете:** Это не сырые сигналы ЭЭГ, а **статистические признаки**, извлечённые методом Jordan Bird et al. (IEEE 2018).
        Каждая колонка — это агрегированная характеристика за эпоху.
        """)

        with st.expander("📖 Расшифровка признаков (нажмите чтобы развернуть)", expanded=False):
            for base, info in feature_groups.items():
                cols_str = ', '.join(info['cols'][:4])
                if len(info['cols']) > 4:
                    cols_str += f' ... и ещё {len(info["cols"]) - 4}'
                st.markdown(
                    f'<div class="feature-group">'
                    f'<div class="feature-group-title">{base}</div>'
                    f'<div class="feature-group-desc">{info["desc"]}</div>'
                    f'<div class="feature-list">Колонки: {cols_str}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<div class="kaggle-section">Визуализация признаков</div>', unsafe_allow_html=True)

        group_names = list(feature_groups.keys())
        if group_names:
            selected_group = st.selectbox(
                "Выберите группу признаков:",
                group_names,
                format_func=lambda x: f"{x} — {feature_groups[x]['desc'][:50]}..."
            )

            if selected_group:
                group_info = feature_groups[selected_group]
                available_cols = group_info['cols']

                st.caption(f"**{group_info['desc']}** — доступно {len(available_cols)} признаков")

                selected_features = st.multiselect(
                    "Выберите конкретные признаки для отображения:",
                    available_cols,
                    default=available_cols[:min(4, len(available_cols))],
                    help="Можно выбрать любое количество"
                )

                if selected_features:
                    fig, ax = plt.subplots(figsize=(14, 5))
                    sns.set_theme(style="ticks")
                    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_features)))

                    for i, (col, color) in enumerate(zip(selected_features, colors)):
                        offset = i * 3
                        values = df[col][:min(200, len(df))].values
                        ax.plot(values + offset, label=col, linewidth=1.5, alpha=0.85, color=color)

                    ax.set_title(f"{group_info['desc']} (эпоха 200 отсчётов)", 
                                fontsize=13, fontweight='bold', pad=12, color='#1A1A1A')
                    ax.set_xlabel("Номер эпохи", fontsize=10, color='#555')
                    ax.set_ylabel("Значение признака + смещение", fontsize=10, color='#555')
                    ax.legend(loc='upper right', frameon=True, fancybox=False, fontsize=8,
                             title="Признаки", title_fontsize=9)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#CCC')
                    ax.spines['bottom'].set_color('#CCC')
                    ax.tick_params(colors='#555')
                    ax.set_facecolor('#FFFFFF')
                    fig.patch.set_facecolor('#FFFFFF')
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown(f'<div class="kaggle-caption">Группа: {selected_group} — {group_info["desc"]}</div>', unsafe_allow_html=True)

    # ─── TAB 2: CLASSIFICATION ─────────────────────────────────────────────
    with tab_analys:
        st.markdown('<div class="kaggle-section">Машинная классификация</div>', unsafe_allow_html=True)
        st.write("Система извлекает случайную запись из данных и отправляет вектор признаков на классификатор.")

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            classify_btn = st.button("▶ Запустить классификацию", width="stretch")

        if classify_btn:
            rand_idx = np.random.randint(0, len(df))
            row = df.iloc[rand_idx]
            # Удаляем только целевую метку (Label / label)
            features_series = row.drop(['Label', 'label'], errors='ignore')
            features_list = [float(val) for val in features_series.values]

            # Гарантируем ровно 989 признаков для модели
            if len(features_list) < 989:
                features_list.extend([0.0] * (989 - len(features_list)))
            elif len(features_list) > 989:
                features_list = features_list[:989]

            mean_val = float(np.mean(features_list))
            std_val = float(np.std(features_list))

            with st.spinner("Запрос к классификатору..."):
                try:
                    response = requests.post(f"{api_url}/predict", json={"features": features_list}, timeout=5)

                    if response.status_code == 200:
                        res = response.json()
                        pred_state = res.get("prediction", "Unknown")
                        confidence = res.get("confidence", 0.0)
                        model_used = res.get("model_used", "ML Model")
                        class_idx = res.get("class_index", 0)
                        probabilities = res.get("probabilities", {})

                        if ANALYTICS_AVAILABLE and hasattr(EEGInterpreter, 'get_clinical_report'):
                            report = EEGInterpreter.get_clinical_report(pred_state)
                        else:
                            report = fallback_clinical_report(pred_state)

                        badge_class = f"badge-{report.get('color', 'neutral')}"
                        block_class = f"info-block-{report.get('color', 'normal')}"

                        st.markdown("---")

                        col_res1, col_res2 = st.columns([1, 2])

                        with col_res1:
                            st.markdown('<div class="kaggle-section">Результат</div>', unsafe_allow_html=True)
                            st.markdown(f'<span class="badge {badge_class}">{report["status_ru"]}</span>', unsafe_allow_html=True)
                            st.write("")

                            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#20BEFF;">{confidence*100:.2f}%</div><div class="metric-label">Уверенность модели</div></div>', unsafe_allow_html=True)

                            st.write("")
                            st.markdown("**Метрики объекта:**")
                            st.markdown(f"- Модель: `{model_used}`")
                            st.markdown(f"- Индекс класса: `{class_idx}`")
                            st.markdown(f"- Среднее вектора: `{mean_val:.4f}`")
                            st.markdown(f"- Стд. откл. вектора: `{std_val:.4f}`")

                            if probabilities:
                                st.write("")
                                st.markdown("**Распределение вероятностей:**")
                                for k, v in probabilities.items():
                                    st.progress(float(v), text=f"{k}: {v*100:.1f}%")

                        with col_res2:
                            st.markdown('<div class="kaggle-section">Клиническое заключение</div>', unsafe_allow_html=True)
                            st.markdown(f'<div class="info-block {block_class}"><strong>Физиологическое резюме:</strong><br>{report["summary"]}</div>', unsafe_allow_html=True)

                            st.markdown("**Рекомендации:**")
                            for rec in report.get("recommendations", []):
                                st.markdown(f"- {rec}")

                    else:
                        st.error(f"Ошибка сервера: HTTP {response.status_code} — {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Ошибка соединения. Убедитесь, что api.py запущен.")
                except requests.exceptions.Timeout:
                    st.error("⏱ Превышено время ожидания.")

    # ─── TAB 3: RHYTHMS ────────────────────────────────────────────────────
    with tab_rhythms:
        st.markdown('<div class="kaggle-section">Спектральный анализ ритмов</div>', unsafe_allow_html=True)
        st.write("Оценка мощности спектральной плотности в стандартных частотных диапазонах (метод Уэлча).")

        col_btn2, _ = st.columns([1, 3])
        with col_btn2:
            rhythm_btn = st.button("▶ Рассчитать ритмы", key="calc_rhythms", width="stretch")

        if rhythm_btn:
            if not ANALYTICS_AVAILABLE:
                st.warning("⚠️ Модуль analytics.py не обнаружен. Выполняется упрощенная демонстрация спектра.")
            else:
                with st.spinner("Вычисление спектральной мощности..."):
                    try:
                        rhythm_df = EEGRhythmAnalyzer.analyze_channels(df, fs=fs)
                        mean_rhythms = EEGRhythmAnalyzer.get_mean_rhythms(df, fs=fs)
                        summary = EEGRhythmAnalyzer.get_rhythm_summary(df, fs=fs)

                        st.markdown("---")
                        st.markdown('<div class="kaggle-section">Мощность по диапазонам</div>', unsafe_allow_html=True)

                        r1, r2, r3, r4, r5 = st.columns(5)
                        rhythm_data = [
                            ('Дельта', 'delta', '0.5–4 Гц', '#8B5CF6'),
                            ('Тета', 'theta', '4–8 Гц', '#6366F1'),
                            ('Альфа', 'alpha', '8–13 Гц', '#10B981'),
                            ('Бета', 'beta', '13–30 Гц', '#3B82F6'),
                            ('Гамма', 'gamma', '30–45 Гц', '#F59E0B'),
                        ]

                        for col, (label, key, band, color) in zip([r1, r2, r3, r4, r5], rhythm_data):
                            with col:
                                val = mean_rhythms.get(key, 0.0)
                                st.markdown(f'<div class="metric-card" style="border-top: 3px solid {color};"><div style="font-size:0.75rem; color:#666; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">{label}</div><div style="font-size:0.65rem; color:#999; margin-bottom:0.5rem;">{band}</div><div class="metric-value" style="font-size:1.5rem;">{val:.4f}</div><div class="metric-label">мкВ²/Гц</div></div>', unsafe_allow_html=True)

                        st.write("")

                        dom = str(summary.get('dominant_rhythm', 'N/A')).upper()
                        dom_val = summary.get('dominant_value', 0.0)
                        total = summary.get('total_power', 0.0)
                        st.markdown(f'<div class="info-block" style="background:#F0F7FF; border-left-color:#3B82F6;"><strong>Доминантный ритм:</strong> <span style="font-size:1.1rem; font-weight:700; color:#1E40AF;">{dom}</span> (мощность {dom_val} мкВ²/Гц) · Общая мощность: <strong>{total:.4f}</strong> мкВ²/Гц</div>', unsafe_allow_html=True)

                        st.markdown("---")

                        st.markdown('<div class="kaggle-section">Мощность по каналам</div>', unsafe_allow_html=True)
                        st.dataframe(rhythm_df.style.format("{:.6f}").background_gradient(cmap='YlGnBu', axis=1), width="stretch", height=350)

                        st.markdown("---")

                        st.markdown('<div class="kaggle-section">Средняя спектральная мощность</div>', unsafe_allow_html=True)
                        fig2, ax2 = plt.subplots(figsize=(10, 4.5))
                        fig2.patch.set_facecolor('#FFFFFF')
                        ax2.set_facecolor('#FFFFFF')
                        labels = [r[0] for r in rhythm_data]
                        values = [mean_rhythms.get(r[1], 0.0) for r in rhythm_data]
                        colors = [r[3] for r in rhythm_data]

                        bars = ax2.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
                        ax2.set_ylabel("Мощность (мкВ²/Гц)", fontsize=10, color='#555')
                        ax2.set_title("Средняя спектральная мощность по всем каналам", fontsize=12, fontweight='bold', pad=10, color='#1A1A1A')
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_color('#CCC')
                        ax2.spines['bottom'].set_color('#CCC')
                        ax2.tick_params(colors='#555')
                        ax2.grid(axis='y', alpha=0.3, color='#E0E0E0')

                        for bar in bars:
                            height = bar.get_height()
                            ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#333')

                        plt.tight_layout()
                        st.pyplot(fig2)

                        st.markdown('<div class="kaggle-section">Тепловая карта</div>', unsafe_allow_html=True)
                        fig3, ax3 = plt.subplots(figsize=(10, max(4, len(rhythm_df)*0.45)))
                        fig3.patch.set_facecolor('#FFFFFF')
                        ax3.set_facecolor('#FFFFFF')
                        sns.heatmap(rhythm_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=ax3, cbar_kws={'label': 'мкВ²/Гц', 'shrink': 0.8}, linewidths=0.5, linecolor='#E0E0E0')
                        ax3.set_title("Спектральная мощность: каналы × частотные диапазоны", fontsize=12, fontweight='bold', pad=12, color='#1A1A1A')
                        ax3.set_xlabel("Частотный диапазон", fontsize=10, color='#555')
                        ax3.set_ylabel("Канал ЭЭГ", fontsize=10, color='#555')
                        ax3.tick_params(colors='#555')
                        plt.tight_layout()
                        st.pyplot(fig3)

                    except Exception as e:
                        st.error(f"Ошибка анализа: {str(e)}")
                        st.exception(e)

else:
    st.markdown('<div style="text-align:center; padding:4rem 2rem; color:#888;"><div style="font-size:3rem; margin-bottom:1rem;">📂</div><div style="font-size:1.2rem; font-weight:600; color:#555; margin-bottom:0.5rem;">Данные не загружены</div><div style="font-size:0.9rem;">Загрузите CSV-файл или поместите mental-state.csv в папку проекта.</div></div>', unsafe_allow_html=True)
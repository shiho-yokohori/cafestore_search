import pandas as pd
import numpy as np
import re
import plotly.express as px
import streamlit as st # ★この行がStreamlitアプリに必須です！★

# Streamlitアプリのタイトルと説明
st.title("カフェ検索＆フィルタリングアプリ")
st.markdown("このアプリを使って、条件に合うカフェを検索・分析できます。")
st.markdown("---") # 区切り線

# --- 1. shop.csv と detail.csv の読み込み ---
@st.cache_data # データロード処理をキャッシュし、アプリの再実行を高速化
def load_data():
    try:
        shop_df = pd.read_csv('shop.csv', sep=',', encoding='utf-8')
        detail_df = pd.read_csv('detail.csv', sep=',', encoding='utf-8')
        return shop_df, detail_df
    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}。ファイルが同じディレクトリにあるか確認してください。")
        st.stop() # エラーでアプリを停止

shop_df, detail_df = load_data()

st.header("データ準備と前処理")

# --- 3. DataFrameの結合（URLの正規化をさらに強化） ---
if 'title_URL' in shop_df.columns and 'title_URL' in detail_df.columns:
    
    def super_normalize_url(url_str):
        if pd.isna(url_str):
            return np.nan
        url_str = str(url_str).strip()
        url_str = url_str.replace('http://', 'https://')
        url_str = url_str.replace('r.tabelog.com', 'tabelog.com')
        if url_str.endswith('/'):
            url_str = url_str[:-1]
        
        match = re.match(r'(https://tabelog\.com/[^/]+/[^/]+/[^/]+/[^/]+)(?:/.*)?', url_str)
        if match:
            url_str = match.group(1)
        else:
            url_str = url_str.replace('/dtl', '').replace('/rvw', '').replace('/map', '')
            url_str = re.sub(r'[?#].*$', '', url_str)
            
        return url_str.lower()

    shop_df.loc[:, 'normalized_url'] = shop_df['title_URL'].apply(super_normalize_url)
    detail_df.loc[:, 'normalized_url'] = detail_df['title_URL'].apply(super_normalize_url)

    cafe_df = pd.merge(detail_df, shop_df, on='normalized_url', how='inner', suffixes=('_detail', '_shop'))

    st.write(f"データ結合が完了しました。合計行数: {len(cafe_df)}")

else:
    st.error("致命的なエラー: 共通の結合キー 'title_URL' が、どちらかのDataFrameに見つかりません。")
    st.stop()


# --- 4. 'bookmark' 列の数値化と最終確認 ---
if not cafe_df.empty and 'bookmark' in cafe_df.columns:
    cleaned_bookmark = cafe_df['bookmark'].astype(str).str.strip().str.replace('人', '', regex=False)
    numeric_bookmark = pd.to_numeric(cleaned_bookmark, errors='coerce')
    
    try:
        cafe_df['bookmark'] = pd.Series(numeric_bookmark.to_numpy(), dtype='Int64', index=cafe_df.index)
    except Exception as e:
        st.warning(f"警告: 'bookmark' 列を Int64 に変換できませんでした: {e}。代わりに float64 で強制的に変換します。")
        cafe_df['bookmark'] = pd.Series(numeric_bookmark.to_numpy(), dtype='float64', index=cafe_df.index)

    if cafe_df['bookmark'].dtype == 'object':
        st.warning("致命的警告: 'bookmark' 列がまだobject型です。最終手段としてfloat64に再々強制変換します。")
        cafe_df['bookmark'] = pd.to_numeric(cafe_df['bookmark'], errors='coerce').astype('float64')
else:
    st.error("'cafe_df' が空であるか、または 'bookmark' 列が見つからないため、'pop_score' の計算はスキップされます。")
    st.stop()


# --- 5. 'star' 列の数値化と 'pop_score' の計算 ---
if not pd.api.types.is_numeric_dtype(cafe_df['star']):
    st.write("star 列を数値化します。")
    cafe_df['star'] = pd.to_numeric(cafe_df['star'], errors='coerce').astype('float64')
    
if pd.api.types.is_numeric_dtype(cafe_df['star']) and pd.api.types.is_numeric_dtype(cafe_df['bookmark']):
    cafe_df['bookmark_for_log'] = cafe_df['bookmark'].fillna(0).apply(lambda x: max(x, 0))
    cafe_df['pop_score'] = cafe_df['star'] * np.log1p(cafe_df['bookmark_for_log'])
    cafe_df = cafe_df.drop(columns=['bookmark_for_log'])
    st.write("'pop_score' 列の計算が完了しました。")
else:
    st.error("エラー: 'star' または 'bookmark' 列が数値型ではないため、'pop_score' を計算できませんでした。")
    st.stop()


# --- 6. 'price' 列の数値化 ---
cleaned_price = cafe_df['price'].astype(str).str.strip().str.replace('円', '', regex=False)
cleaned_price = cleaned_price.str.replace(',', '', regex=False)

numeric_price = pd.to_numeric(cleaned_price, errors='coerce')

try:
    cafe_df['price'] = pd.Series(numeric_price.to_numpy(), dtype='Int64', index=cafe_df.index)
except Exception as e:
    st.warning(f"警告: 'price' 列を Int64 に変換できませんでした: {e}。代わりに float64 で強制的に変換します。")
    cafe_df['price'] = pd.Series(numeric_price.to_numpy(), dtype='float64', index=cafe_df.index)

if cafe_df['price'].dtype == 'object':
    st.warning("致命的警告: 'price' 列がまだobject型です。最終手段としてfloat64に再々強制変換します。")
    cafe_df['price'] = pd.to_numeric(cafe_df['price'], errors='coerce').astype('float64')


# --- 'comment' 列の数値化処理を追加 ---
cafe_df['comment_count'] = cafe_df['comment'].astype(str).str.extract(r'(\d+)', expand=False)
cafe_df['comment_count'] = pd.to_numeric(cafe_df['comment_count'], errors='coerce')


# cafe_df の pop_score と price などの統計量確認
with st.expander("データ概要と統計量を見る"): # 折りたたみ可能なセクション
    st.subheader("主要な数値列の統計量")
    if 'pop_score' in cafe_df.columns:
        st.dataframe(cafe_df[['pop_score', 'price', 'star', 'bookmark', 'comment_count']].describe())
    else:
        st.write("'pop_score' 列がまだ存在しません。")


# --- フィルタリングとソートのためのサイドバー設定 ---
st.sidebar.header("検索条件")

# 利用可能なフィルタの定義
available_filters = {
    '人気スコア': {'column': 'pop_score', 'type': float, 'prompt_suffix': ' (推奨: 5.0以上)'},
    '星の評価': {'column': 'star', 'type': float, 'prompt_suffix': ' (推奨: 3.0以上)'},
    'コメント数': {'column': 'comment_count', 'type': int, 'prompt_suffix': ' (推奨: 10以上)'},
    'ブックマーク数': {'column': 'bookmark', 'type': int, 'prompt_suffix': ' (推奨: 100以上)'}
}

# Streamlitのマルチセレクタでフィルタを選択
selected_filter_names = st.sidebar.multiselect(
    "適用したいフィルタを選択してください:",
    options=list(available_filters.keys()),
    default=[]
)

filter_conditions = {}
if selected_filter_names:
    st.sidebar.subheader("各フィルタの最低値を設定")
    for filter_name in selected_filter_names:
        info = available_filters[filter_name]
        
        # 該当カラムの現在の最小値・最大値を取得 (NaNを除外)
        col_data = cafe_df[info['column']].dropna()
        if not col_data.empty:
            min_val = col_data.min()
            max_val = col_data.max()
        else: # データがない場合、デフォルト値を設定
            min_val = 0.0 if info['type'] is float else 0
            max_val = 1000.0 if info['type'] is float else 10000

        # Streamlitの数値入力ウィジェットを使用
        if info['type'] is float:
            value = st.sidebar.number_input(
                f"{filter_name}の最低値{info['prompt_suffix']}:",
                min_value=float(min_val),
                max_value=float(max_val) if pd.notna(max_val) else 1000.0,
                value=float(min_val), # 初期値は現在の最低値
                step=0.1,
                format="%.1f",
                key=f"filter_{info['column']}"
            )
        else: # intの場合
            value = st.sidebar.number_input(
                f"{filter_name}の最低値{info['prompt_suffix']}:",
                min_value=int(min_val),
                max_value=int(max_val) if pd.notna(max_val) else 10000,
                value=int(min_val), # 初期値は現在の最低値
                step=1,
                format="%d",
                key=f"filter_{info['column']}"
            )
        filter_conditions[info['column']] = value
else:
    st.sidebar.write("フィルタリング条件は設定されていません。")


st.sidebar.markdown("---") # サイドバーの区切り線

# --- ソート機能のためのサイドバー設定 ---
sortable_columns = ['pop_score', 'star', 'bookmark', 'price', 'comment_count']

selected_sort_columns = st.sidebar.multiselect(
    "ソートする項目を選択してください (選択順に優先):",
    options=sortable_columns,
    default=[]
)

ascending_list = []
if selected_sort_columns:
    st.sidebar.subheader("各ソート項目の順序:")
    for col in selected_sort_columns:
        # radioボタンの選択肢とキーを設定
        sort_order_choice = st.sidebar.radio(
            f"{col} のソート順:",
            options=["降順 (大きい順)", "昇順 (小さい順)"],
            index=0, # デフォルトは降順
            key=f"sort_order_{col}"
        )
        ascending_list.append(True if sort_order_choice == "昇順 (小さい順)" else False)


# --- フィルタ処理 ---
filtered_df = cafe_df.copy() # ここで必ず filtered_df を初期化！

if filter_conditions:
    st.subheader("適用されたフィルタ")
    for col, value in filter_conditions.items():
        filter_name = next((name for name, info in available_filters.items() if info['column'] == col), col)
        if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]):
            filtered_df = filtered_df[filtered_df[col] >= value]
            st.write(f"- **{filter_name}** が **{value}** 以上")
        else:
            st.warning(f"警告: '{col}' 列が見つからないか数値型ではないため、'{filter_name}' フィルタは適用されません。")
else:
    st.subheader("フィルタは適用されていません")


if filtered_df.empty: 
    st.error("フィルタリング条件に合うカフェは見つかりませんでした。条件を緩めてください。")
else: 
    st.success(f"フィルタリング後のカフェ数: **{len(filtered_df)}** 件")
    
    # --- ソートの適用 ---
    if selected_sort_columns and not filtered_df.empty:
        st.subheader("ソート結果（上位10件）")
        try:
            filtered_df_sorted = filtered_df.sort_values(
                by=selected_sort_columns,
                ascending=ascending_list,
                na_position='last'
            )
            st.dataframe(filtered_df_sorted[['title_shop','star','comment','comment_count','pop_score','price','seats','bookmark', 'access']].head(10)) 
        except KeyError as e:
            st.error(f"ソート項目 '{e}' が見つからないか、データ型が適切ではありません。")
    else:
        st.subheader("フィルタリング結果（上位10件）")
        st.dataframe(filtered_df[['title_shop','star','comment','comment_count','pop_score','price','seats','bookmark', 'access']].head(10))


# --- プロットの作成と表示 ---
st.header("人気スコアとブックマーク数の散布図")
if not filtered_df.empty and \
   ('pop_score' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['pop_score'])) and \
   ('bookmark' in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df['bookmark'])):
    try:
        fig = px.scatter(
            filtered_df,
            x='pop_score',
            y='bookmark',
            hover_data=['title_shop', 'star', 'comment', 'comment_count', 'price', 'seats', 'access'], 
            title='人気スコアとブックマーク数の散布図'
        )
        st.plotly_chart(fig) # StreamlitでPlotlyの図を表示
    except Exception as e:
        st.error(f"散布図の表示中にエラーが発生しました: {e}")
else:
    st.write("散布図を表示するための十分なデータ、または適切な列がありません。")

st.markdown("---")
st.write("データ処理が完了しました。")
import pandas as pd
import numpy as np
import re
import os

# --- 1. 文件设置 ---
INPUT_FILE = "camera_data.json"
OUTPUT_FILE = "camera_data_clean3.csv"

# --- 2. 清洗辅助函数：将文本转换为纯数值 ---

def clean_weight(weight_str):
    """提取重量，统一转换为克 (g)"""
    if pd.isna(weight_str): return np.nan
    weight_str = str(weight_str).lower().replace(',', '').strip()
    # 查找数字，默认单位为克
    match = re.search(r'([\d.]+)', weight_str)
    return float(match.group(1)) if match else np.nan

def clean_max_iso(iso_str):
    """提取 ISO 范围中的最大值"""
    if pd.isna(iso_str): return np.nan
    # 查找所有数字，取最后一个（通常是最大值）
    numbers = re.findall(r'(\d+)', str(iso_str).replace(',', '').strip())
    return float(numbers[-1]) if numbers else np.nan

def clean_aperture_f(aperture_str):
    """【新列】提取最大光圈（F值最小的数值）"""
    if pd.isna(aperture_str): return np.nan
    # 查找所有数字，取最小值
    numbers = [float(n) for n in re.findall(r'[\d.]+', str(aperture_str)) if n]
    return min(numbers) if numbers else np.nan

def clean_shutter_speed(shutter_str):
    """统一快门速度，转换为秒数"""
    if pd.isna(shutter_str): return np.nan
    shutter_str = str(shutter_str).lower().strip()
    
    # 处理分数形式（如 1/1000 sec）
    if '/' in shutter_str and 'sec' in shutter_str:
        parts = shutter_str.split('/')
        try:
            return 1.0 / float(re.findall(r'(\d+)', parts[-1])[0])
        except:
            return np.nan
    # 处理整数形式（如 30 sec）
    match = re.search(r'([\d.]+)', shutter_str)
    return float(match.group(1)) if match else np.nan

def clean_exposure_range(exp_str):
    """提取曝光补偿的最大范围数值 (EV)"""
    if pd.isna(exp_str): return np.nan
    # 匹配 ±X EV 模式中的 X 数值
    match = re.search(r'[\u00b1+-]\s*([\d.]+)\s*EV', str(exp_str))
    return float(match.group(1)) if match else np.nan

def clean_screen_res(res_str):
    """提取屏幕分辨率（点数）"""
    if pd.isna(res_str): return np.nan
    # 提取第一个数字作为点数
    numbers = re.findall(r'(\d+)', str(res_str).replace(',', '').lower())
    return float(numbers[0]) if numbers else np.nan

def clean_screen_size(size_str):
    """提取屏幕尺寸 (英寸)"""
    if pd.isna(size_str): return np.nan
    return pd.to_numeric(str(size_str).replace('"', '').strip(), errors='coerce')


def clean_focus_range(focus_str):
    """提取对焦距离，统一转换为厘米 (cm)"""
    if pd.isna(focus_str): return np.nan
    focus_str = str(focus_str).lower().replace('cm', '').strip()
    
    # 将 m 转换为 cm
    match_m = re.search(r'([\d.]+)m', focus_str)
    if match_m: return float(match_m.group(1)) * 100 
        
    # 直接提取 cm 数值
    match_cm = re.search(r'([\d.]+)', focus_str)
    return float(match_cm.group(1)) if match_cm else np.nan

def clean_dimensions(dim_str):
    """从 LxWxH 字符串中提取 L, W, H"""
    if pd.isna(dim_str): return (np.nan, np.nan, np.nan)
    
    # 使用严格正则匹配整数或浮点数
    numbers = [float(n) for n in re.findall(r'\d+\.\d+|\d+', str(dim_str))]
    
    # 确保提取了至少三个数字 (L, W, H)
    return (numbers[0], numbers[1], numbers[2]) if len(numbers) >= 3 else (np.nan, np.nan, np.nan)


# --- 3. 核心清洗函数 ---

def clean_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """执行全部清洗、重命名和派生指标计算"""
    print(f"--- 1. 开始清洗 {len(df)} 条数据 ---")
    
    # --- A. 清洗并覆盖原始列为数值 ---
    
    # 将多列的文本数据直接转换为数值 (覆盖原列)
    df['Weight'] = df['Weight'].apply(clean_weight)
    df['ISO'] = df['ISO'].apply(clean_max_iso)
    df['Min. shutter speed'] = df['Min. shutter speed'].apply(clean_shutter_speed)
    df['Max. shutter speed'] = df['Max. shutter speed'].apply(clean_shutter_speed)
    df['Exposure Compensation'] = df['Exposure Compensation'].apply(clean_exposure_range)
    df['Screen resolution'] = df['Screen resolution'].apply(clean_screen_res)
    df['Screen size'] = df['Screen size'].apply(clean_screen_size)
    df['Normal focus range'] = df['Normal focus range'].apply(clean_focus_range)
    df['Macro focus range'] = df['Macro focus range'].apply(clean_focus_range)

    # Max aperture：保留原列，新增数值列
    df['Min_Aperture_F'] = df['Max aperture'].apply(clean_aperture_f) 
    
    # 重命名已清洗的列
    df.rename(columns={
        'Weight': 'Weight_g',
        'ISO': 'Max_ISO',
        'Min. shutter speed': 'Min_Shutter_Speed_Sec',
        'Max. shutter speed': 'Max_Shutter_Speed_Sec',
        'Exposure Compensation': 'Max_Exposure_Comp',
        'Screen resolution': 'Screen_Res_Dots',
        'Screen size': 'Screen_Size_in',
        'Normal focus range': 'Normal_Focus_cm',
        'Macro focus range': 'Macro_Focus_cm',
    }, inplace=True)
    
    # 尺寸分离 (创建 L, W, H 三个新列)
    df[['Dim_L', 'Dim_W', 'Dim_H']] = df['Dimensions'].apply(lambda x: pd.Series(clean_dimensions(x)))
    
    # 视频支持：检查 Max. video resolution 是否包含 4K 关键词
    df['Supports_4K'] = df['Max. video resolution'].astype(str).str.contains('3840|4096|4K', case=False, na=False)

    # 整理其他通用数值列
    for col in ['Total megapixels', 'Effective megapixels', 'Megapixels', 'Crop factor']:
         df[col] = pd.to_numeric(df[col], errors='coerce')


    print("--- 2. 生成派生指标 (推荐系统评分维度) ---")
    
    # --- B. 派生指标计算 (使用百分位数排名进行标准化) ---
    
    # 1. 便携性评分 (重量 60% + 体积 40%)
    df['Portability_Score'] = (
        df['Weight_g'].rank(ascending=False, na_option='bottom', pct=True) * 0.6 + 
        (1 / (df['Dim_L'] * df['Dim_W'] * df['Dim_H'])).rank(ascending=True, na_option='bottom', pct=True) * 0.4
    ) * 100
    df['Portability_Score'] = df['Portability_Score'].clip(20, 100)
    
    # 2. 弱光性能评分 (ISO 50% + 光圈 30% + 裁切系数 20%)
    df['Aperture_Value'] = 1 / df['Min_Aperture_F'] 
    df['LowLight_Score'] = (
        df['Max_ISO'].rank(pct=True, na_option='top') * 0.5 + 
        df['Aperture_Value'].rank(pct=True, na_option='top') * 0.3 +
        df['Crop factor'].rank(pct=True, na_option='top', ascending=False) * 0.2
    ) * 100
    df['LowLight_Score'] = df['LowLight_Score'].clip(20, 100)
    
    # 3. 视频性能评分 (4K 支持 40% + 屏幕尺寸 30% + 最快快门速度 30%)
    df['Video_Score'] = (
        df['Supports_4K'].astype(int) * 0.4 + 
        df['Screen_Size_in'].rank(pct=True, na_option='top') * 0.3 +
        df['Max_Shutter_Speed_Sec'].rank(pct=True, na_option='top') * 0.3
    ) * 100
    df['Video_Score'] = df['Video_Score'].clip(20, 100)
    
    # 调整 Min_Aperture_F 列的位置，紧跟在 Max aperture 之后
    cols = df.columns.tolist()
    aperture_idx = cols.index('Max aperture')
    if 'Min_Aperture_F' in cols:
        cols.remove('Min_Aperture_F') 
        cols.insert(aperture_idx + 1, 'Min_Aperture_F')
        df = df[cols] 

    return df

# --- 4. 主执行函数 ---

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}。请确保它与脚本在同一目录下。")
        return

    try:
        df = pd.read_json(INPUT_FILE) 
    except Exception as e:
        print(f"加载 JSON 文件失败，请检查文件格式是否正确。错误: {e}")
        return

    df_clean = clean_and_derive_features(df)
    
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"\n--- 成功！数据已清洗并保存到当前目录: {OUTPUT_FILE} ---")

if __name__ == "__main__":
    main()
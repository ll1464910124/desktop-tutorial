import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tushare as ts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置页面
st.set_page_config(page_title="股票多指标决策系统", page_icon="📈", layout="wide")

class AdvancedTradingDecisionSystem:
    def __init__(self, token):
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
    def get_stock_basic_info(self, ts_code):
        """获取股票基本信息"""
        try:
            df = self.pro.stock_basic(ts_code=ts_code, 
                                     fields='ts_code,symbol,name,area,industry,list_date')
            if not df.empty:
                return df.iloc[0]['name']
            return None
        except:
            return None
        
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据"""
        try:
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df is None or df.empty:
                return None
            df = df.sort_values('trade_date')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df.set_index('trade_date', inplace=True)
            return df
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return None

    def get_sector_performance(self, start_date, end_date):
        """获取板块表现数据"""
        try:
            # 获取所有股票的基本信息
            stocks = self.pro.stock_basic(exchange='', list_status='L', 
                                         fields='ts_code,name,industry,area')
            
            # 获取主要行业板块
            sectors = {
                '金融': ['银行', '保险', '证券'],
                '科技': ['软件服务', '互联网', '半导体', '通信设备', '信息技术'],
                '消费': ['食品饮料', '家用电器', '商贸零售', '旅游酒店'],
                '医药': ['医药制造', '医疗保健', '生物制品'],
                '制造': ['机械设备', '汽车制造', '电气设备', '国防军工'],
                '周期': ['有色金属', '煤炭', '钢铁', '化工', '建筑材料'],
                '地产': ['房地产开发', '房地产服务'],
                '能源': ['电力', '石油', '天然气'],
                '交通': ['交通运输', '物流', '航空机场'],
                '公用事业': ['公用事业', '环保工程']
            }
            
            # 获取最近一个月的交易日
            trade_cal = self.pro.trade_cal(exchange='', start_date=start_date, end_date=end_date)
            trade_days = trade_cal[trade_cal['is_open'] == 1]['cal_date'].tolist()
            
            sector_data = {}
            
            for sector_name, industries in sectors.items():
                # 找到属于该板块的股票
                sector_stocks = stocks[stocks['industry'].isin(industries)]['ts_code'].tolist()
                
                if not sector_stocks:
                    continue
                
                # 获取板块内股票数据
                sector_performance = []
                for stock in sector_stocks[:50]:  # 限制数量避免请求过多
                    try:
                        stock_data = self.pro.daily(ts_code=stock, start_date=start_date, end_date=end_date)
                        if stock_data is not None and not stock_data.empty:
                            # 计算个股表现
                            first_day = stock_data.iloc[0]
                            last_day = stock_data.iloc[-1]
                            change_pct = (last_day['close'] - first_day['close']) / first_day['close'] * 100
                            avg_volume = stock_data['vol'].mean()
                            avg_amount = stock_data['amount'].mean()
                            
                            sector_performance.append({
                                'ts_code': stock,
                                'change_pct': change_pct,
                                'avg_volume': avg_volume,
                                'avg_amount': avg_amount
                            })
                    except:
                        continue
                
                if sector_performance:
                    sector_df = pd.DataFrame(sector_performance)
                    sector_data[sector_name] = {
                        'stock_count': len(sector_performance),
                        'avg_change': sector_df['change_pct'].mean(),
                        'total_volume': sector_df['avg_volume'].sum(),
                        'total_amount': sector_df['avg_amount'].sum(),
                        'up_ratio': len(sector_df[sector_df['change_pct'] > 0]) / len(sector_df) * 100
                    }
            
            return sector_data
            
        except Exception as e:
            st.error(f"获取板块数据失败: {e}")
            return None

    def get_index_data(self, index_codes):
        """获取指数数据"""
        index_data = {}
        for code in index_codes:
            try:
                df = self.pro.index_daily(ts_code=code, 
                                         start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                                         end_date=datetime.now().strftime('%Y%m%d'))
                if df is not None and not df.empty:
                    df = df.sort_values('trade_date')
                    first_close = df.iloc[0]['close']
                    last_close = df.iloc[-1]['close']
                    change_pct = (last_close - first_close) / first_close * 100
                    index_data[code] = {
                        'name': self.get_index_name(code),
                        'change_pct': change_pct,
                        'current': last_close
                    }
            except:
                continue
        return index_data

    def get_index_name(self, index_code):
        """获取指数名称"""
        index_names = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000300.SH': '沪深300',
            '000905.SH': '中证500',
            '399005.SZ': '中小板指'
        }
        return index_names.get(index_code, index_code)

    def calculate_macd(self, df, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        df = df.copy()
        df['EMA_fast'] = df['close'].ewm(span=fast).mean()
        df['EMA_slow'] = df['close'].ewm(span=slow).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=signal).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # 计算MACD斜率和DEA斜率
        df['MACD_slope'] = df['MACD'].diff()
        df['DEA_slope'] = df['MACD_signal'].diff()
        
        return df

    def calculate_ma_system(self, df):
        """计算均线系统"""
        df = df.copy()
        df['MA5'] = df['close'].rolling(5).mean()
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA60'] = df['close'].rolling(60).mean()
        df['MA120'] = df['close'].rolling(120).mean()
        
        # 计算均线方向
        df['MA20_direction'] = df['MA20'].diff()
        df['MA60_direction'] = df['MA60'].diff()
        df['MA120_direction'] = df['MA120'].diff()
        
        return df

    def calculate_rsi(self, df, periods=[6, 12, 24]):
        """计算RSI指标（多周期）"""
        df = df.copy()
        for period in periods:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return df

    def calculate_kdj(self, df, n=9, m1=3, m2=3):
        """计算KDJ指标"""
        df = df.copy()
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        
        df['RSV'] = (df['close'] - low_min) / (high_max - low_min) * 100
        df['K'] = df['RSV'].ewm(alpha=1/m1).mean()
        df['D'] = df['K'].ewm(alpha=1/m2).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        # 添加前一日数据用于金叉死叉判断
        df['K_prev'] = df['K'].shift(1)
        df['D_prev'] = df['D'].shift(1)
        return df

    def calculate_bollinger_bands(self, df, period=20, std=2):
        """计算布林带"""
        df = df.copy()
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # 计算布林带位置
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df

    def calculate_volume_indicators(self, df):
        """计算成交量指标"""
        df = df.copy()
        df['VMA5'] = df['vol'].rolling(5).mean()
        df['VMA20'] = df['vol'].rolling(20).mean()
        df['volume_ratio'] = df['vol'] / df['VMA5']
        
        # 计算OBV
        df['OBV'] = (np.sign(df['close'].diff()) * df['vol']).fillna(0).cumsum()
                
        # 计算OBV趋势
        df['OBV_trend'] = df['OBV'].diff()
        return df

    def calculate_atr(self, df, period=14):
        """计算ATR"""
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(period).mean()
        return df

    def calculate_cci(self, df, period=14):
        """计算CCI"""
        df = df.copy()
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['CCI'] = (tp - sma) / (0.015 * mad)
        return df

    def calculate_sar(self, df, acceleration=0.02, maximum=0.2):
        """计算SAR指标"""
        df = df.copy()
        high = df['high'].values
        low = df['low'].values
        sar = np.zeros(len(df))
        trend = np.zeros(len(df))
        af = acceleration
        ep = low[0]
        hp = high[0]
        lp = low[0]
        
        sar[0] = low[0] - (high[0] - low[0]) * 0.1
        trend[0] = 1 if sar[0] < low[0] else -1
        
        for i in range(1, len(df)):
            if trend[i-1] < 0:
                sar[i] = sar[i-1] - af * (sar[i-1] - hp)
                if high[i] > hp:
                    af = min(af + acceleration, maximum)
                    hp = high[i]
                if sar[i] < low[i]:
                    trend[i] = -1
                else:
                    trend[i] = 1
                    sar[i] = lp
                    af = acceleration
                    lp = low[i]
            else:
                sar[i] = sar[i-1] + af * (lp - sar[i-1])
                if low[i] < lp:
                    af = min(af + acceleration, maximum)
                    lp = low[i]
                if sar[i] > high[i]:
                    trend[i] = 1
                else:
                    trend[i] = -1
                    sar[i] = hp
                    af = acceleration
                    hp = high[i]
        
        df['SAR'] = sar
        df['SAR_trend'] = trend
        return df

    def calculate_additional_indicators(self, df):
        """计算更多专业指标"""
        df = df.copy()
        
        # 威廉指标
        period = 14
        df['WR'] = (df['high'].rolling(period).max() - df['close']) / (df['high'].rolling(period).max() - df['low'].rolling(period).min()) * -100
        
        # DMI指标
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['+DM'] = np.where(
            (df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
            np.maximum(df['high'] - df['high'].shift(1), 0), 0
        )
        df['-DM'] = np.where(
            (df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
            np.maximum(df['low'].shift(1) - df['low'], 0), 0
        )
        
        # 计算14日平滑
        df['TR_14'] = df['TR'].rolling(14).mean()
        df['+DM_14'] = df['+DM'].rolling(14).mean()
        df['-DM_14'] = df['-DM'].rolling(14).mean()
        
        # 计算DI
        df['+DI'] = (df['+DM_14'] / df['TR_14']) * 100
        df['-DI'] = (df['-DM_14'] / df['TR_14']) * 100
        
        # 计算ADX
        dx = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
        df['ADX'] = dx.rolling(14).mean()
        
        # 资金流向指标 (MFI)
        df['MFI'] = self.calculate_mfi(df)
        
        return df

    def calculate_mfi(self, df, period=14):
        """计算资金流向指标"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['vol']
        
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(period).sum()
        negative_mf = pd.Series(negative_flow).rolling(period).sum()
        
        money_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi

    def calculate_all_indicators(self, df):
        """计算所有技术指标"""
        if df is None or len(df) < 60:
            return df
            
        df = self.calculate_macd(df)
        df = self.calculate_ma_system(df)
        df = self.calculate_rsi(df)
        df = self.calculate_kdj(df)
        df = self.calculate_bollinger_bands(df)
        df = self.calculate_volume_indicators(df)
        df = self.calculate_atr(df)
        df = self.calculate_cci(df)
        df = self.calculate_sar(df)
        df = self.calculate_additional_indicators(df)
        
        return df

    def data_quality_check(self, df):
        """数据质量检查 - 只检查原始数据，不检查技术指标"""
        if df is None or len(df) == 0:
            return {
                'has_issues': True,
                'issues': ['数据为空或无效'],
                'data_quality_score': 0
            }
            
        issues = []
        
        # 只检查原始数据字段
        original_columns = ['open', 'high', 'low', 'close', 'vol']
        
        # 检查原始数据缺失值
        missing_data = df[original_columns].isnull().sum()
        if missing_data.any():
            issues.append(f"原始数据存在缺失: {dict(missing_data[missing_data > 0])}")
        
        # 检查异常值
        price_change = df['close'].pct_change().abs()
        outlier_days = price_change[price_change > 0.1]  # 单日涨跌幅超过10%
        if len(outlier_days) > 0:
            issues.append(f"发现{len(outlier_days)}个价格异常交易日")
        
        # 检查成交量异常
        volume_outliers = df[df['vol'] == 0]
        if len(volume_outliers) > 0:
            issues.append(f"发现{len(volume_outliers)}个零成交量交易日")
        
        # 检查数据连续性
        date_diff = df.index.to_series().diff().dt.days
        gap_days = date_diff[date_diff > 1]
        if len(gap_days) > 0:
            issues.append(f"发现{len(gap_days)}个数据断点")
        
        # 技术指标的NaN是正常的，不视为问题
        score_deduction = len([issue for issue in issues if "原始数据" in issue or "异常" in issue or "零成交量" in issue])
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'data_quality_score': max(0, 100 - score_deduction * 20)
        }

# ... 其他类保持不变 (RiskManagementSystem, MarketSentimentAnalyzer, BacktestingEngine, TradingDecisionEngine) ...

# 在display函数部分添加新的板块热度展示函数
def display_sector_heatmap(analyzer):
    """显示板块资金热度图"""
    st.subheader("🔥 A股板块资金热度图")
    
    # 计算日期范围（最近一个月）
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
    
    with st.spinner("正在获取板块资金流向数据..."):
        try:
            # 获取板块表现数据
            sector_data = analyzer.get_sector_performance(start_date, end_date)
            
            if not sector_data:
                st.warning("无法获取板块数据，请检查网络连接或Token权限")
                return
            
            # 转换为DataFrame
            sector_df = pd.DataFrame(sector_data).T
            sector_df = sector_df.reset_index().rename(columns={'index': '板块'})
            
            # 计算热度分数（综合考虑涨跌幅、成交额、上涨股票比例）
            sector_df['热度分数'] = (
                sector_df['avg_change'] * 0.4 + 
                (sector_df['total_amount'] / sector_df['total_amount'].max() * 100) * 0.4 +
                sector_df['up_ratio'] * 0.2
            )
            
            # 排序
            sector_df = sector_df.sort_values('热度分数', ascending=False)
            
            # 创建热力图
            st.write("#### 📊 板块资金热度排行榜")
            
            # 使用Plotly创建交互式热力图
            fig = go.Figure(data=go.Heatmap(
                z=[sector_df['热度分数'].values],
                x=sector_df['板块'].values,
                y=['热度'],
                colorscale='RdYlGn',
                showscale=True,
                hoverongaps=False,
                hovertemplate='板块: %{x}<br>热度: %{z:.1f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='A股板块资金热度图 (越红代表越热)',
                xaxis_title='板块',
                yaxis_title='',
                height=400,
                xaxis=dict(tickangle=45)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示详细数据表格
            st.write("#### 📈 板块详细数据")
            
            # 格式化显示数据
            display_df = sector_df[['板块', 'stock_count', 'avg_change', 'total_amount', 'up_ratio', '热度分数']].copy()
            display_df['平均涨跌幅'] = display_df['avg_change'].apply(lambda x: f"{x:.2f}%")
            display_df['总成交额(亿)'] = (display_df['total_amount'] / 100000000).apply(lambda x: f"{x:.2f}")
            display_df['上涨比例'] = display_df['up_ratio'].apply(lambda x: f"{x:.1f}%")
            display_df['热度分数'] = display_df['热度分数'].apply(lambda x: f"{x:.1f}")
            display_df['股票数量'] = display_df['stock_count']
            
            final_df = display_df[['板块', '股票数量', '平均涨跌幅', '总成交额(亿)', '上涨比例', '热度分数']]
            
            # 使用颜色渐变显示
            def color_heatmap(val, column):
                if column == '平均涨跌幅':
                    value = float(val.replace('%', ''))
                    if value > 0:
                        return f"background-color: rgba(255, 0, 0, {min(0.3 + value/50, 0.8)})"
                    else:
                        return f"background-color: rgba(0, 255, 0, {min(0.3 + abs(value)/50, 0.8)})"
                elif column == '热度分数':
                    value = float(val)
                    intensity = min(value / 100, 1)
                    return f"background-color: rgba(255, 0, 0, {0.2 + intensity * 0.6})"
                return ""
            
            styled_df = final_df.style.applymap(
                lambda x: color_heatmap(x, '平均涨跌幅'), 
                subset=['平均涨跌幅']
            ).applymap(
                lambda x: color_heatmap(x, '热度分数'), 
                subset=['热度分数']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # 显示主要指数表现
            st.write("#### 📋 主要指数表现")
            index_codes = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH', '000905.SH']
            index_data = analyzer.get_index_data(index_codes)
            
            if index_data:
                index_list = []
                for code, data in index_data.items():
                    index_list.append({
                        '指数': data['name'],
                        '涨跌幅': f"{data['change_pct']:.2f}%",
                        '当前点位': f"{data['current']:.2f}",
                        '状态': '📈' if data['change_pct'] > 0 else '📉'
                    })
                
                index_df = pd.DataFrame(index_list)
                st.dataframe(index_df, use_container_width=True)
            
            # 添加分析结论
            st.write("#### 💡 板块热度分析")
            
            top_sectors = sector_df.head(3)
            bottom_sectors = sector_df.tail(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**最热板块**:")
                for _, sector in top_sectors.iterrows():
                    st.write(f"- **{sector['板块']}**: 热度{sector['热度分数']:.1f} (涨{sector['avg_change']:.2f}%)")
            
            with col2:
                st.write("**最冷板块**:")
                for _, sector in bottom_sectors.iterrows():
                    st.write(f"- **{sector['板块']}**: 热度{sector['热度分数']:.1f} (涨{sector['avg_change']:.2f}%)")
            
            # 投资建议
            st.write("#### 🎯 投资建议")
            hottest_sector = sector_df.iloc[0]
            st.info(f"""
            **当前市场热点**: {hottest_sector['板块']}板块
            - 热度评分: {hottest_sector['热度分数']:.1f}
            - 平均涨幅: {hottest_sector['avg_change']:.2f}%
            - 资金关注度: 非常高
            
            **操作建议**: 
            - 关注{hottest_sector['板块']}板块的龙头个股
            - 注意热点轮动，避免追高风险
            - 结合技术指标选择合适买入时机
            """)
            
        except Exception as e:
            st.error(f"生成板块热度图时出现错误: {e}")
            st.info("这可能是由于API限制或网络问题导致，请稍后重试")

def display_sector_analysis(analyzer):
    """显示板块分析页面"""
    st.header("🏢 A股板块资金热度分析")
    
    st.markdown("""
    ### 板块资金热度说明
    
    本模块展示最近一个月A股各板块的资金流向和热度情况，帮助您识别市场热点：
    
    - **🔥 热度分数**: 综合考量板块涨跌幅、成交额、上涨股票比例
    - **📈 平均涨跌幅**: 板块内股票的平均价格变化
    - **💰 总成交额**: 板块总资金流入规模
    - **📊 上涨比例**: 板块内上涨股票占比
    
    **颜色说明**: 越红色代表资金热度越高，越绿色代表相对冷清
    """)
    
    # 添加刷新按钮
    if st.button("🔄 刷新板块数据", type="primary"):
        st.rerun()
    
    # 显示板块热度图
    display_sector_heatmap(analyzer)
    
    # 添加时间说明
    st.write(f"---")
    st.caption(f"数据更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("数据来源: Tushare | 分析周期: 最近30个交易日")

# ... 其他display函数保持不变 (display_price_charts, display_technical_indicators_table, display_decision_analysis, display_indicator_details, display_data_quality_report, display_risk_management_report, display_market_sentiment, display_backtest_results) ...

def main():
    st.title("🎖️ 股票多指标决策系统")
    st.markdown("基于**分层指挥体系**的智能交易决策平台")
    
    # 侧边栏配置
    st.sidebar.header("配置参数")
    
    # Tushare token输入
    token = st.sidebar.text_input("Tushare API Token", type="password", 
                                 help="请在Tushare官网注册获取API Token")
    
    if not token:
        st.warning("请输入Tushare API Token以继续")
        st.info("""
        **如何获取Tushare Token:**
        1. 访问 [Tushare官网](https://tushare.pro) 注册账号
        2. 在个人中心获取API Token
        3. 将Token粘贴到左侧输入框中
        
        **示例股票代码:**
        - 000001.SZ (平安银行)
        - 600000.SH (浦发银行)
        - 000858.SZ (五粮液)
        """)
        return
    
    # 初始化分析器
    analyzer = AdvancedTradingDecisionSystem(token)
    
    # 股票代码输入
    ts_code = st.sidebar.text_input("股票代码", "000001.SZ", 
                                   help="格式：代码.交易所，如000001.SZ, 600000.SH")
    
    # 自动获取股票名称
    stock_name = "未知股票"
    if ts_code:
        with st.spinner("正在获取股票信息..."):
            name = analyzer.get_stock_basic_info(ts_code)
            if name:
                stock_name = name
                st.sidebar.success(f"股票名称: {stock_name}")
            else:
                st.sidebar.warning("未能自动获取股票名称，请检查代码格式")
    
    # 日期范围选择
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        start_date_input = st.text_input("开始日期", start_date)
    with col4:
        end_date_input = st.text_input("结束日期", end_date)
    
    # 获取数据
    if st.sidebar.button("开始分析", type="primary"):
        with st.spinner("正在获取数据并计算指标..."):
            try:
                # 获取股票数据
                df = analyzer.get_stock_data(ts_code, start_date_input, end_date_input)
                
                if df is None or df.empty:
                    st.error("未能获取到股票数据，请检查股票代码和日期范围")
                    st.info("""
                    **可能的原因:**
                    1. 股票代码格式错误
                    2. Token无效或过期
                    3. 选择的日期范围内无交易数据
                    4. 网络连接问题
                    """)
                    return
                
                if len(df) < 60:
                    st.warning(f"数据长度较短（{len(df)}个交易日），部分长期指标可能不准确")
                
                # 计算所有技术指标
                df_with_indicators = analyzer.calculate_all_indicators(df)
                
                if df_with_indicators is None or len(df_with_indicators) == 0:
                    st.error("计算技术指标失败，数据不足")
                    return
                
                # 显示基本信息
                st.subheader(f"🎯 {stock_name} ({ts_code}) 多指标决策分析")
                
                # 显示分层指挥体系说明
                with st.expander("🎖️ 分层指挥体系说明", expanded=True):
                    st.write("""
                    ### 分层指挥体系 - 优先级铁律
                    
                    | 类别        | **作战任务**        | **主/辅级别**     | **使用场景** | **信号权重** |
                    | :-------- | :-------------- | :------------ | :------- | :------- |
                    | **趋势指标**  | **定方向**（能不能做）   | **主帅**（最高优先级） | 日线以上周期   | **50%**  |
                    | **成交量指标** | **验真伪**（是不是骗）   | **政委**（一票否决制） | 所有场景     | **30%**  |
                    | **动量指标**  | **找时机**（何时进出）   | **参谋**（辅助确认）  | 60分钟-日线  | **15%**  |
                    | **波动率指标** | **划边界**（目标位/止损） | **工兵**（技术支撑）  | 入场后管理    | **5%**   |
                    
                    **优先级铁律**: 
                    - 趋势指标定仓位（50%+还是空仓）
                    - 成交量定是否入场（达标才执行）
                    - 动量指标定买卖点（精细优化）
                    """)
                
                # 获取决策分数用于风险管理
                decision_engine = TradingDecisionEngine()
                if len(df_with_indicators) >= 2:
                    current_data = df_with_indicators.iloc[-1]
                    prev_data = df_with_indicators.iloc[-2]
                    decision_scores = decision_engine.evaluate_conditions(current_data, prev_data)
                    signal_strength = decision_scores['total_score']
                else:
                    signal_strength = 50
                
                # 创建标签页 - 新增板块分析标签
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "📈 价格走势", "📊 技术指标", "🤖 决策分析", "🔍 指标详解",
                    "🔍 数据质量", "🛡️ 风险管理", "😊 市场情绪", "📊 回测分析", "🏢 板块热度"
                ])
                
                with tab1:
                    display_price_charts(df_with_indicators, stock_name)
                
                with tab2:
                    display_technical_indicators_table(df_with_indicators)
                
                with tab3:
                    display_decision_analysis(df_with_indicators)
                
                with tab4:
                    display_indicator_details(df_with_indicators)
                    
                with tab5:
                    display_data_quality_report(df_with_indicators, analyzer)
                
                with tab6:
                    display_risk_management_report(df_with_indicators, signal_strength)
                
                with tab7:
                    display_market_sentiment(df_with_indicators, analyzer)
                
                with tab8:
                    display_backtest_results(df_with_indicators)
                
                with tab9:
                    display_sector_analysis(analyzer)
                    
            except Exception as e:
                st.error(f"分析过程中出现错误: {e}")
                st.info("""
                **常见问题解决方法:**
                1. 检查Tushare Token是否正确
                2. 确认股票代码格式正确（如：000001.SZ）
                3. 尝试调整日期范围
                4. 检查网络连接
                """)
    
    # 如果没有点击开始分析，直接显示板块热度
    else:
        display_sector_analysis(analyzer)

if __name__ == "__main__":
    main()

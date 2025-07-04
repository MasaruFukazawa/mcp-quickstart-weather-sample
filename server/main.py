from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import validate_call


#  FastMCP サーバの初期化
mcp = FastMCP("weather")


WEATHER_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"


@validate_call
async def make_nws_request(url: str) -> dict[str, Any] | None:
    """ 適切なエラーハンドリングでNWS APIにリクエストを出す
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


@validate_call
def format_alert(feature: dict) -> str:
    """アラート機能を読みやすい文字列に整形する
    """
    props = feature["properties"]

    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""


@validate_call
@mcp.tool()
async def get_alerts(state: str) -> str:
    """米国各州の天気予報アラートを受け取る

    Args:
        state: 2文字の米国州コード（例：CA、NY）

    Return:
        天気予報アラート文字列
    """
    url = f"{WEATHER_API_BASE}/alerts/active/area/{state}"
    response_data = await make_nws_request(url)

    if not response_data or "features" not in response_data:
        return "アラートを取得できない、またはアラートが見つからない"

    elif not response_data["features"]:
        return "この状態に対するアクティブなアラートはありません"

    alerts = [format_alert(feature) for feature in response_data["features"]]

    return "\n---\n".join(alerts)


@validate_call
@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """場所の天気予報を取得します

    Args:
        latitude: 緯度
        longitude: 経度
    """
    # 予測グリッドの終点を取得する
    points_url = f"{WEATHER_API_BASE}/points/{latitude},{longitude}"
    points_response_data = await make_nws_request(points_url)

    if not points_response_data:
        return "この場所の予報データを取得できません"

    # ポイントレスポンスから予想URLを取得
    forecast_url = points_response_data["properties"]["forecast"]
    forecast_response_data = await make_nws_request(forecast_url)

    if not forecast_response_data:
        return "詳細な予測を取得できない"

    # 期間を読みやすい予測にフォーマットする
    periods = forecast_response_data["properties"]["periods"]

    forecasts = []

    for period in periods[:5]:  # 次の5つのデータのみ表示
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)


if __name__ == "__main__":

    # サーバーの初期化と実行
    mcp.run(transport='stdio')

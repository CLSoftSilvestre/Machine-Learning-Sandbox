import openmeteo_requests
import requests_cache
from retry_requests import retry

class WeatherConnector:

    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    def __init__(self, lat=38.8259, lon=-25.5094):
        self.url = "http://api.open-meteo.com/v1/forecast"
        self.lat = lat
        self.lon = lon

        self.params = {
            "latitude": lat,
            "longitude": lon,
            "current": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"],
            "timezone": "auto",
            "forecast_days": 1
        }
    
    def GetData(self):
        self.responses = self.openmeteo.weather_api(self.url, params= self.params)
        return self.responses[0].Current()

    


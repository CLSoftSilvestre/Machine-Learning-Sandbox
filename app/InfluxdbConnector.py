import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

class InfluxDBPoint:
    def __init__(self, point, tag, field):
        self.point = point
        self.tag = tag
        self.field = field
    
    def SetValue(self, value):
        self.value = value

class InfluxDBConnector:
    def __init__(self, bucket, organization, token, url):
        self.bucket = bucket
        self.organization = organization
        self.token = token
        self.url = url
        self.points = []
        self.client = influxdb_client.InfluxDBClient(
            url = self.url,
            token = self.token,
            org=self.organization
        )
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def AddPoint(self, point=InfluxDBPoint):
        self.points.append(point)

    def WritePoints(self):     
        for point in self.points:
            p = influxdb_client.Point(point.point).tag("tag",point.tag).field(point.field, point.value)
            self.write_api(bucket=self.bucket, org=self.org, record=p)
import urllib3
import json

class TeamsWebhookException(Exception):
    """custom exception for failed webhook call"""
    pass

class TeamsConnector:
    def __init__(self, hookurl, http_timeout=60, cooldown=60, operator="lower", target=0):
        self.http = urllib3.PoolManager()
        self.payload = {}
        self.hookurl = hookurl
        self.http_timeout = http_timeout
        self.cooldown = int(cooldown)
        self.remainingCooldown = 0
        self.operator = operator
        self.target = float(target)
    
    def text(self, mtext:str, value):
        replacedText = mtext.replace('{VALUE}', str(value))
        self.payload["text"] = replacedText
        self.testConditions(value)
        return
    
    def cooldownReduction(self, amount=1):
        self.remainingCooldown -= int(amount)
        return
    
    def testConditions(self, value):
        try:
            if self.operator == "lower":
                if value < self.target:
                    self.send()
            elif self.operator == "equal":
                if value == self.target:
                    self.send()
            elif self.operator == "higher":
                if value > self.target:
                    self.send()
            else:
                pass
        except Exception as err:
            print(str(err))

    def send(self):
        if self.remainingCooldown <= 0:
            headers = {"Content-Type":"application/json"}
            r = self.http.request(
                    'POST',
                    f'{self.hookurl}',
                    body=json.dumps(self.payload).encode('utf-8'),
                    headers=headers, timeout=self.http_timeout)
            if r.status == 200:
                self.remainingCooldown = int(self.cooldown)
                return True
            else:
                raise TeamsWebhookException(r.reason)
        return False
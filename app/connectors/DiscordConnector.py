import pip_system_certs.wrapt_requests
from discord_webhook import DiscordWebhook

class DiscordWebhookException(Exception):
    """custom exception for failed webhook call"""
    pass

class DiscordConnector:
    def __init__(self, hookurl, cooldown=60, operator="lower", target=0):
        self.hookurl = hookurl
        self.cooldown = int(cooldown)
        self.operator = operator
        self.target = float(target)
        self.remainingCooldown = 0
        self.message = None
        #self.title = None
        #self.color = "03b2f8"
    
    def text(self, mtext:str, value):
        pos = mtext.find("|")
        roundDigits = 2
        tempText = mtext
        if pos > -1:
            roundDigits = int(mtext[pos+1])
            strToRemove = "|" + str(mtext[pos+1])
            tempText = tempText.replace(strToRemove,"")

        replacedText = tempText.replace('{VALUE}', str(round(value, roundDigits)))
        self.message = replacedText
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
            webhook = DiscordWebhook(url=self.hookurl, content=self.message)
            response = webhook.execute()
            self.remainingCooldown = int(self.cooldown)
        return False
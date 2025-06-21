class Player:
  def __init__(self, number, money, bot):
    self.BOT = bot
    self.money = int(money)
    self.onTable = 0

  def setMoney(self, money):
    self.money = int(money)
  
  def getMoney(self):
    return self.money
  
  def isBot(self):
    return self.BOT
  
  def getOnTable(self):
    return self.onTable

  def setOnTable(self, money):
    self.onTable = money  
table = 0

def bidUp(amount):
  if (amount > table):
    table = amount
    return True
  return False

def getTable():
  return table
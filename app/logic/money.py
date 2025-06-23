table = 0

def bidUp(amount):
  global table
  if (amount >= table and amount != 0):
    table = amount
    return True
  return False

def getTable():
  return table
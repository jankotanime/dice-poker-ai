from collections import Counter

def pointCount(dices):
  if None in dices:
    return 0
  actPoints = 0
  if {1, 2, 3, 4, 5}.issubset(dices) or {2, 3, 4, 5, 6}.issubset(dices):
    actPoints += 13+sum(dices)/100
  else:
    dice = Counter(dices)
    for i in dice.keys():
      actPoints += int(i)**(1/10)*(dice[i]**2)
  return actPoints
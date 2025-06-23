from ai.functions import *
from logic.score import *
from logic.money import getTable
from game.move import *
import random
import pickle; 


def roll_dices(n):
  dices = []
  for _ in range (n):
    dices.append(random.randint(1, 7))
  return dices

def apply_mask_and_reroll(dices, best_mask_opt):
    kept_dices = [dice for dice, keep in zip(dices, best_mask_opt) if keep == 0]
    reroll_count = best_mask_opt.count(1)
    new_dices = roll_dices(reroll_count)
    return kept_dices + new_dices

def ai_first_tour(player):
  table = getTable()
  money = player.getMoney()

  if (table <= money):
    amount = table
  else:
    amount = money
  player.setMoney(player.getMoney() - amount)
  print("AI stawia:", amount)

  dices = roll_dices(5)
  score = pointCount(dices)

  player.setDices(dices)
  player.setScore(score)

  print("AI rzuca kości:", dices)
  print("AI zdobywa punkty:", score)

def ai_second_tour(player, opponent):
  clf = pickle.load(open("model/play-model/bet/model-1.pkl", "rb")); 

  table = getTable()
  dices = player.getDices()
  money = player.getMoney()
  score = player.getScore()

  print(dices)

  best_mask_opt, best_score_opt = best_mask_and_score(dices)

  bet = clf.predict([[money, opponent.getMoney(), table, best_score_opt, opponent.getScore()]])

  if (not bet[0]):
    return False
  
  if (table <= money):
    amount = table
  else:
    amount = money
  player.setMoney(player.getMoney() - amount)
  print("AI stawia:", amount)
  print("AI maska:", best_mask_opt)
  
  dices = apply_mask_and_reroll(dices, best_mask_opt)
  score = pointCount(dices)

  player.setDices(dices)
  player.setScore(score)

  print("AI rzuca kości:", dices)
  print("AI zdobywa punkty:", score)

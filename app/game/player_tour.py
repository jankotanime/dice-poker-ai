from ai.recognise_dices import process_image
from logic.score import *
from logic.money import getTable, bidUp
from game.move import *

def player_first_tour(player):
  table = getTable()
  money = player.getMoney()
  if (not player.isBot()):
    amount = player_move(table, money)
    
    if (not amount):
      return False
    
    print("Postawiona kwota:", amount)
    bidUp(amount)
    player.setMoney(player.getMoney() - amount)

  while True:
    image_number = input("Podaj numer zdjęcia (pamiętaj, że zdjęcie musi być w folderze input-data): ")
    try:
      image_number = int(image_number)
    except ValueError:
      print("Błąd! Podany int")
      continue
    dices = process_image(image_number)
    score = pointCount(dices)
    print("Wyrzucone kości:", dices, "Wynik:", score)
    player.setDices(dices)
    player.setScore(score)
    return True


def player_second_tour(player):
  table = getTable()
  money = player.getMoney()
  print("Twoja kwota:", money)
  print("Twoje kości:", player.getDices())
  print("Twój wynik: ", player.getScore())
  if (not player.isBot()):
    amount = player_move(table, money, True)
    
    if (not amount):
      return False
    
    print("Postawiona kwota:", amount+table)
    bidUp(amount)
    player.setMoney(player.getMoney() - amount)

  while True:
    image_number = input("Podaj numer zdjęcia (pamiętaj, że zdjęcie musi być w folderze input-data): ")
    try:
      image_number = int(image_number)
    except ValueError:
      print("Błąd! Podany int")
      continue
    dices = process_image(image_number)
    score = pointCount(dices)
    print("Wyrzucone kości:", dices, "Wynik:", score)
    return True

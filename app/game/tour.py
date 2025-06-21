from ai.recognise_dices import process_image
from logic.score import *
from logic.money import getTable
from game.move import *
from game import playerslist

def tour(t, playerNumber):
  table = getTable()
  player = playerslist[playerNumber]
  money = player.getMoney()
  if (not player.isBot()):
    amount = player_move(table, money)
    if (amount == None):
      return
    
    if (not amount):
      return
    
    print("Postawiona kwota:", amount)

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
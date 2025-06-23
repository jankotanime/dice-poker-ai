from game.player_tour import player_first_tour, player_second_tour
from game.ai_tour import ai_first_tour, ai_second_tour
from object.player import Player
from logic.money import getTable
import sys

def surr(player):
  if (player.isBot()):
    print("Przygrywasz!")
    print("Twoja kwota:", player.getMoney())
  else:
    print("Wygrywasz!")
    print("Twoja kwota:", player.getMoney()+getTable()*2)

  sys.exit(0)

if __name__ == "__main__":
  player1 = Player(1, 1000, False)
  player2 = Player(2, 1000, True)
  player_first_tour(player1)
  ai_first_tour(player2)
  player_second_tour(player1)
  ai_second_tour(player2, player1)

  if (player1.getScore() == player2.getScore()):
    print("remis!")
    print("Twoja kwota:", player1.getMoney()+getTable())

  elif (player1.getScore() > player2.getScore()):
    print("Wygrywasz!")
    print("Twoja kwota:", player1.getMoney()+getTable()*2)
  else:
    print("Przegrywasz!")
    print("Twoja kwota:", player1.getMoney())
  

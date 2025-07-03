import sys

def player_move(table, money, second_tour = False):
  print("Aktualna stawka:", table)
  print("Twój portfel:", )
  print("q - wyjdź")
  print("p - pass")
  print("v - va baque")
  print("b - podbij")

  while True:
    cmd = input("Co chcesz zrobić?")
    match cmd:
      case 'q':
        sys.exit(0)
      case 'p':
        return 0
      case 'v':
        return money
      case 'b':
        while True:
          amount = input("Podaj kwote: ")
          try:
            amount = int(amount)
          except ValueError:
            print("Błąd! Podany int")
            continue
          if ((isinstance(amount, int) and amount <= money and amount > 0 and amount >= table) or second_tour):
            return amount
          print("Błędna wartość!")
      
  
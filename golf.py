# Golf model: ball flight, conditions, AI caddie, fun game where ai caddie guesses yardage    

import csv
import math
import random
from sklearn.linear_model import LinearRegression

#  CLUBS
# club: [carry yards, launch angle, spin rpm]

clubs = {
    "Driver": [295, 11.5, 2200],
    "2-wood": [280, 13.0, 2900],
    "3-wood": [260, 12, 3650],
    "7-wood": [240, 13.5, 4600],
    "9-wood": [210, 18.0, 5500],
    "6-iron": [190, 14.1, 5730],
    "7-iron": [175, 16.3, 6600],
    "8-iron": [160, 18.1, 7500],
    "9-iron": [150, 20.4, 8000],
    "PW":     [140, 24.2, 8700],
    "GW":     [130, 27.0, 9000],
    "SW":     [120, 30.0, 9750],
    "LW":     [105, 33.0, 10100]
}

# ---------------- HOMECOURSE------------
HOME_ALTITUDE = 60    # feet above sea level at your home course
GREEN_SIZE = 4500       # green area in square feet
GREEN_RADIUS = math.sqrt(GREEN_SIZE / math.pi) / 3   # ~12.6 yards from pin to edge (round green)


# ---------------- BALL FLIGHT ----------------
# Moves the ball forward in tiny time steps until it lands.
# wind: + into you, - behind you   altitude: feet   elevation: + uphill, - downhill
def fly(ball_speed, launch, spin, wind=0, altitude=0, elevation=0):
    air = 1.225 * math.exp(-altitude * 0.3048 / 8500)     # air gets thinner up high
    vx = ball_speed * 0.447 * math.cos(math.radians(launch))
    vy = ball_speed * 0.447 * math.sin(math.radians(launch))
    spin = spin * 0.1047                                   # rpm -> radians per second
    wind = wind * 0.447
    x = 0
    y = 0
    while True:
        rx = vx + wind                                     # ball speed relative to the air
        speed = math.sqrt(rx * rx + vy * vy)
        s = 0.0213 * spin / speed                          # how strong the spin is
        drag = 0.22 + 0.30 * s
        lift = 0.45 * s ** 0.4
        force = air * 0.0156 * speed                       # bigger when faster / thicker air
        vx = vx + (-drag * rx - lift * vy) * force * 0.01
        vy = vy + (-drag * vy + lift * rx) * force * 0.01 - 9.81 * 0.01
        x = x + vx * 0.01
        y = y + vy * 0.01
        if vy < 0 and y <= elevation * 0.3048:             # falling and reached the green
            return x * 1.094                               # meters -> yards


# Find each club's ball speed
ball_speed = {}
for name in clubs:
    carry_yds, launch, spin = clubs[name]
    speed = 60
    while fly(speed, launch, spin, 0, HOME_ALTITUDE, 0) < carry_yds:
        speed = speed + 0.2
    ball_speed[name] = speed


def carry(club, wind=0, elevation=0):
    launch = clubs[club][1]
    spin = clubs[club][2]
    return fly(ball_speed[club], launch, spin, wind, HOME_ALTITUDE, elevation)


def on_green(miss):
    return abs(miss) <= GREEN_RADIUS


# ---------------- AI CADDIE ----------------
# Learns from shot_log.csv: club, carry, wind, elevation
def read_shot(row):
    stock = clubs[row["club"]][0]
    wind = float(row["wind"])
    elevation = float(row["elevation"])
    return stock, [wind, stock * wind, elevation], float(row["carry"])


def train_caddie(rows=None):
    if rows is None:
        rows = list(csv.DictReader(open("shot_log.csv")))
    inputs = []
    answers = []
    for row in rows:
        stock, features, real = read_shot(row)
        inputs.append(features)
        answers.append(real - stock)                 # learn the ADJUSTMENT from your stock yardage
    return LinearRegression().fit(inputs, answers)


def caddie_guess(model, club, wind, elevation):
    stock = clubs[club][0]
    return stock + model.predict([[wind, stock * wind, elevation]])[0]


def caddie_pick(model, target, wind, elevation):
    best_club = None
    best_miss = 9999
    for club in clubs:
        miss = abs(caddie_guess(model, club, wind, elevation) - target)
        if miss < best_miss:
            best_miss = miss
            best_club = club
    return best_club


def stock_pick(target):
    best_club = None
    best_miss = 9999
    for club in clubs:
        if abs(clubs[club][0] - target) < best_miss:
            best_miss = abs(clubs[club][0] - target)
            best_club = club
    return best_club


def conditions_study():
    print("\nChange in carry (yards) vs. no wind, flat, at home")
    print("Club      Carry  10mph into  10mph behind  30ft uphill  30ft downhill")
    for club in clubs:
        base = carry(club)
        head = carry(club, wind=10) - base
        tail = carry(club, wind=-10) - base
        up = carry(club, elevation=30) - base
        down = carry(club, elevation=-30) - base
        print(club.ljust(9), str(round(base)).ljust(6), str(round(head)).ljust(11),
              ("+" + str(round(tail))).ljust(13), str(round(up)).ljust(12), "+" + str(round(down)))



def play():
    model = train_caddie()
    wind = random.randint(-20, 20)                 # conditions are set ONCE per round
    elevation = random.randint(-30, 30)
    print("\nTODAY: wind", wind, "mph (+ = into you), greens", elevation, "ft (+ = uphill)")
    print("Greens are", GREEN_SIZE, "sq ft: land within", round(GREEN_RADIUS, 1), "yards of the pin to hit the green.")
    total = 0
    greens = 0
    for hole in range(1, 10):
        target = random.randint(95, 260)
        tip = caddie_pick(model, target, wind, elevation)
        print("\nHole", hole, "-", target, "yards. Caddie says:", tip)
        club = input("Your club: ").strip()
        while club not in clubs:
            club = input("Pick one of " + ", ".join(clubs) + ": ").strip()
        shot = carry(club, wind, elevation) * random.gauss(1, 0.03)
        miss = shot - target
        total = total + abs(miss)
        if on_green(miss):
            greens = greens + 1
            result = "ON THE GREEN"
        else:
            result = "missed the green"
        print("  Carried", round(shot), "yards ->", round(abs(miss)), "long" if miss > 0 else "short", "-", result)
    print("\nGreens hit:", greens, "of 9 | Total:", round(total), "yards off the pin")


def study():
    model = train_caddie()
    stock_total = 0
    caddie_total = 0
    stock_greens = 0
    caddie_greens = 0
    shots = 0
    print("\nSimulating 300 rounds...")
    for r in range(300):
        wind = random.randint(-20, 20)
        elevation = random.randint(-30, 30)
        for hole in range(9):
            target = random.randint(95, 260)
            swing = random.gauss(1, 0.03)          # same swing for both strategies
            a = carry(stock_pick(target), wind, elevation) * swing - target
            b = carry(caddie_pick(model, target, wind, elevation), wind, elevation) * swing - target
            stock_total = stock_total + abs(a)
            caddie_total = caddie_total + abs(b)
            stock_greens = stock_greens + on_green(a)
            caddie_greens = caddie_greens + on_green(b)
            shots = shots + 1
    print("Stock yardages: average miss", round(stock_total / shots, 1), "yards, greens hit", round(100 * stock_greens / shots), "%")
    print("AI caddie:      average miss", round(caddie_total / shots, 1), "yards, greens hit", round(100 * caddie_greens / shots), "%")


def test_caddie():
    rows = list(csv.DictReader(open("shot_log.csv")))
    random.shuffle(rows)
    cut = int(len(rows) * 0.8)
    model = train_caddie(rows[:cut])
    test_rows = rows[cut:]

    stock_miss = 0
    caddie_miss = 0
    for row in test_rows:
        stock, features, real = read_shot(row)
        guess = stock + model.predict([features])[0]
        stock_miss = stock_miss + abs(stock - real)
        caddie_miss = caddie_miss + abs(guess - real)
    print("\nTrained on", cut, "shots, tested on", len(test_rows), "shots")
    print("Stock yardages missed your real carry by:", round(stock_miss / len(test_rows), 1), "yards")
    print("AI caddie missed your real carry by:     ", round(caddie_miss / len(test_rows), 1), "yards")


# ---------------- MENU ----------------
while True:
    print("\n1 = conditions study   2 = play a round   3 = strategy study   4 = test caddie   q = quit")
    choice = input("> ").strip()
    if choice == "1":
        conditions_study()
    elif choice == "2":
        play()
    elif choice == "3":
        study()
    elif choice == "4":
        test_caddie()
    elif choice == "q":
        break

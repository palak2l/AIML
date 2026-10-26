import matplotlib.pyplot as plt

def smart_home(temp, light, motion):
    fan = "ON" if temp > 30 else "OFF"
    bulb = "ON" if light < 40 and motion else "OFF"
    return fan, bulb

temp = float(input("Temperature: "))
light = int(input("Light level (0-100): "))
motion = int(input("Motion (1=Yes, 0=No): "))

fan, bulb = smart_home(temp, light, motion)

print("Fan:", fan)
print("Light:", bulb)

peas = [
    ["Performance", "Comfort, Energy Saving"],
    ["Environment", "Smart Home"],
    ["Actuators", "Fan, Lights"],
    ["Sensors", "Temperature, Light, Motion"]
]

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
ax.table(cellText=peas, colLabels=["PEAS", "Description"],
         loc="center", cellLoc="center")
plt.title("Smart Home Agent - PEAS")
plt.show()
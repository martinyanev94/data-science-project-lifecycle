class RoboticArm:
    def __init__(self):
        self.position = "ready"
        self.object_present = False

    def pick_up(self):
        if self.object_present:
            self.position = "holding object"
            print("Picked up the object!")
        else:
            print("No object to pick up.")

    def place_down(self):
        if self.position == "holding object":
            self.position = "ready"
            print("Placed the object down.")
        else:
            print("No object in hand to place down.")

    def detect_object(self):
        self.object_present = True
        print("Object detected!")

arm = RoboticArm()
arm.detect_object()
arm.pick_up()
arm.place_down()

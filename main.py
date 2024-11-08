#main
import alice
import bob
import eve

longueur=1000
recv=alice.alice(longueur, False)
basesToSend, measures = bob.bob(recv, longueur)

print(bob.presumably(basesToSend, measures, longueur))
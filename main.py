import alice
import bob
import eve

longueur=50
recv=alice.alice(longueur, False)
basesToSend, measures = bob.bob(recv, longueur)

listbobkey = bob.presumably(basesToSend, measures, longueur)

bobReveal, bobIndex = bob.revealFromBob(listbobkey)

diff = alice.checkSpy(bobReveal, bobIndex)

bobfinalkey = bob.getFinalKey(listbobkey, bobReveal, bobIndex, diff)

print(bobfinalkey)
print(alice.AliceFinalKey)

print(bobfinalkey == alice.AliceFinalKey)
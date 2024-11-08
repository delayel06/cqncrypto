import alice
import bob
import eve
import sys


if __name__ == "__main__":
    longueur=1000
    flag = sys.argv[1]

    if flag == "eve":
        alice_circuits = alice.alice(longueur, False)
        recv = eve.attack_bb84(alice_circuits, longueur)
    else:   
        recv = alice.alice(longueur, False)

    basesToSend, measures = bob.bob(recv, longueur)

    listbobkey = bob.presumably(basesToSend, measures, longueur)

    bobReveal, bobIndex = bob.revealFromBob(listbobkey)

    diff = alice.checkSpy(bobReveal, bobIndex)

    bobfinalkey = bob.getFinalKey(listbobkey, bobReveal, bobIndex, diff)

    if flag != "eve":
        print(bobfinalkey)
        print(alice.AliceFinalKey)
   
        print("Final key is correct : ", bobfinalkey == alice.AliceFinalKey)

    alicedataframe = alice.mapForPandasAlice()
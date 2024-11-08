from matplotlib import pyplot
import alice
import bob
import eve
import sys
import pandas as pd


def checkBasesDataframe(basesofBob):
    global basesstorage
    basesstorage = []

    correctedBases = alice.checkBases(basesofBob)

    for i in range(len(correctedBases)):
        if correctedBases[i]:
            basesstorage.append('/')
        else:
            basesstorage.append(' ')

    
    data = {'id': list(range(1, 1001)), 'Good bases': basesstorage}

    

    return pd.DataFrame(data)



if __name__ == "__main__":
    success = 0
    mistake = 0
    plot_error_rate = []
    for i in range(10):
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

    

#//////////////////////////////////////////////////////////////////////////////////////////////


        alicedata = alice.mapForPandasAlice()
        bobdata = bob.mapForPandasBob()

        dataframe = pd.merge(alicedata, bobdata, on='id')


        goodbasesdata = checkBasesDataframe(basesToSend)
        dataframe2 = pd.merge(dataframe, goodbasesdata, on='id')

        bitsrevealedmapped = bob.PandasBobSentBits(bobReveal, bobIndex)


        dataframe3 = pd.merge(dataframe2, bitsrevealedmapped, on='id')


        print(dataframe3)


        #If the final key is correct, we increment the success counter
        if bobfinalkey == alice.AliceFinalKey:
            success += 1
        else:
            mistake +=1
            error_rate = 0
            for i in range(bobfinalkey):
                if bobfinalkey[i] != alice.AliceFinalKey[i]:
                    error_rate += 1
            error_rate = error_rate / len(bobfinalkey)
    #Line graph or error rate over time
    # 
    plot_error_rate.append(error_rate)
    pyplot.plot(plot_error_rate)


    # Plot histogram of the success rate, and the mistake rate
    x = ['Success', 'Mistake']
    y = [success, mistake]
    pyplot.bar(x, y)
    pyplot.show()


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


def main(): 
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

        print("Final key is correct : ", bobfinalkey == alice.AliceFinalKey)
    print ((bobfinalkey))
    print ((alice.AliceFinalKey))


#//////////////////////////////////////////////////////////////////////////////////////////////


    alicedata = alice.mapForPandasAlice()
    bobdata = bob.mapForPandasBob()

    dataframe = pd.merge(alicedata, bobdata, on='id')


    goodbasesdata = checkBasesDataframe(basesToSend)
    dataframe2 = pd.merge(dataframe, goodbasesdata, on='id')

    bitsrevealedmapped = bob.PandasBobSentBits(bobReveal, bobIndex)


    dataframe3 = pd.merge(dataframe2, bitsrevealedmapped, on='id')


    #print(dataframe3)

   
    

        #If the final key is correct, we increment the success counte
        
    return bobfinalkey ,alice.AliceFinalKey



if __name__ == "__main__":
    
    success = 0
    fail = 0

    outBob = [False] * 10
    outAlice = [False] * 10

    for i in range(10) : 
        outBob[i], outAlice[i] = main()


    for i in range(len(outBob)) : 
        error_rate = 0
        if outBob[i]==outAlice[i] : 
            success += 1
        else : 
            fail += 1
            
            for j in range(len(outBob[i])):
                if outBob[i][j] != outAlice[i][j]:
                    error_rate += 1
            error_rate =100* (float(error_rate)) / float(len(outBob[i]))  
        plot_error_rate.append(error_rate)
    #Line graph or error rate over time
    # 
    
    
    pyplot.plot(plot_error_rate)
     # Plot histogram of the success rate, and the mistake rate
    x = ['Success', 'Mistake']
    y = [success, fail]
    #pyplot.bar(x, y)
    pyplot.show()
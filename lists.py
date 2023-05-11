def width_values():
    '''
    SET-UP 1
    '''
    #A (=clean) -- length = 4 
    list1A = ["A1","A2","A3","A4"]
    list_width_1A = [0.02485, 0.02498, 0.2490, 0.02496]

    #B (=clean+UV[7min]) -- length = 4
    list1B = ["B1","B2","B3","B4"]
    list_width_1B = [0.02498, 0.02501, 0.02497, 0.02495]

    #C (=clean+sanding+clean) -- length = 5
    list1C = ["C1","C2","C3","C4","C5"]
    list_width_1C = [0.02503, 0.02492, 0.02503, 0.02501, 0.02483]

    #D (=clean+sanding+clean+UV[7min]) -- length = 4
    list1D = ["D1","D2","D3","D4"]
    list_width_1D = [0.02501, 0.2493, 0.02484, 0.02494]

    '''
    SET-UP 2
    '''
    #SP (=smooth patterning) -- length = 4
    list2SP = ['SP1', 'SP2', 'SP3', 'SP4', 'SP5']
    list_width_2SP = [0.02498, 0.02495, 0.02496, 0.02502, 0.02492]

    #RP (=rough patterning) -- length = 4
    list2RP = ['RP1', 'RP2', 'RP3', 'RP4', 'RP5']
    list_width_2RP = [0.02490, 0.02494, 0.02497, 0.02494, 0.02497]


    '''
    SET-UP 3 -- Sandblasting patterning
    '''
    #SBP (=Sandblasting patterning) -- length = 2
    listSBP = ['SBP1', 'SBP2']
    list_width_SBP = [0.02502, 0.02499]

    #PP (=Peelply patterning) -- length = 2
    listPP = ['PP1', 'PP2']
    list_width_PP = [0.02486, 0.02489]


    '''
    SET-UP 4 -- Peelply strip patterning
    '''
    #1D Patterning
    list1DS = ['1DS1', '1DS2', '1DS3', '1DS4', '1DS5']
    list_width_1DS = [0.02493, 0.02490, 0.02490, 0.02485, 0.02480]

    #2D Patterning
    list2D = ['2DS1', '2DS2', '2DS3', '2DS4', '2DS5']
    list_width_2D = [0.02473, 0.02497, 0.02475, 0.02484, 0.02493]

    list_specimens = list1A + list1B + list1C + list1D + list2D + list2SP + list2RP + listPP + listSBP + list1DS

    list_widths = list_width_1A + list_width_1B + list_width_1C + list_width_1D + list_width_2D + list_width_2SP + list_width_2RP + list_width_PP + list_width_SBP + list_width_1DS

    res = {list_specimens[i]: list_widths[i] for i in range(len(list_specimens))}
    return res

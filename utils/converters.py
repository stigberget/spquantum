from . import const

def convert_length(X,unit,target='m'):
    """
    Args:

    @X : int, float, double, or numpy array 

    @unit : assigned unit of length for the passed variable X. Permitted values are {'m','mm','mum','nm','angstrom'}

    @target : length scale to be converted. Permitted values are {'m','mm','mum','nm','angstrom'}

    Returns:

    X : int, float, double, or numpy array converted to the target unit

    target : unit of length of X

    """

    scaleIDs = {'m':1,'mm':1e-3,'mum':1e-6,'nm':1e-9,'angstrom':1e-10}

    if(unit not in scaleIDs.keys()):
        KeyError("The specified unit was not recognized - choose valid unit")
    elif(target not in scaleIDs.keys()):
        KeyError("The specified target unit was not recognized - choose a valid target unit")
    else: 
        scale = scaleIDs[unit]/scaleIDs[target]
        X = X*scale
    
    return X,target

def convert_energy(X,unit,target='J'):

    scaleIDs = {'J':1,'eV':const.enull}

    if(unit not in scaleIDs.keys()):
        KeyError("The specified unit was not recognized - choose valid unit")
    elif(target not in scaleIDs.keys()):
        KeyError("The specified target unit was not recognized - choose a valid target unit")
    else: 
        scale = scaleIDs[unit]/scaleIDs[target]
        X = X*scale
    
    return X,target

def convert_potential(X,unit,target='V'):


    scaleIDs = {'V':1,'mV':1e-3,'eV':1/const.enull,'meV':1e-3/const.enull}

    if(unit not in scaleIDs.keys()):
        KeyError("The specified unit was not recognized - choose valid unit")
    elif(target not in scaleIDs.keys()):
        KeyError("The specified target unit was not recognized - choose a valid target unit")
    else: 
        scale = scaleIDs[unit]/scaleIDs[target]
        X = X*scale
    
    return X,target

def convert_mass(X,unit,target='kg'):
    scaleIDs = {'kg':1,'mg':1e-3,'ng':1e-9,'erm':const.m_electron,'mel':const.m_electron}

    if(unit not in scaleIDs.keys()):
        KeyError("The specified unit was not recognized - choose valid unit")
    elif(target not in scaleIDs.keys()):
        KeyError("The specified target unit was not recognized - choose a valid target unit")
    else: 
        scale = scaleIDs[unit]/scaleIDs[target]
        X = X*scale
    
    return X,target

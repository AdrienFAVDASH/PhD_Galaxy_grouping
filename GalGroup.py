from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import c
from astropy.cosmology import LambdaCDM
from PyAstronomy import pyasl
from sklearn.neighbors import KDTree
from tqdm.notebook import tqdm
import matchGAMA

def main(dataframe, zmin=0.002, zmax=0.65, mag_col='R_PETRO', Z_col='Z_TONRY', Kcorr_col='KCORR_R', vel_err_col='VEL_ERR', RA_col='RA', Dec_col='DEC', ID_col='CATAID', H0=100, Om0=0.25, Ode0=0.75):
    
    #list of constants used in this code
    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    h = cosmo.H(0) / 100
    alpha=-1.26 #Table 4 of Loveday et al. 2015
    M_star=-20.71 #Table 4 of Loveday et al. 2015
    phi_star=10**(-2.02) #Table 4 of Loveday et al. 2015
    b0 = 0.06 #Table 1 of Robotham et al. 2011
    R0 = 18 #Table 1 of Robotham et al. 2011
    Eb = -0.00 #Table 1 of Robotham et al. 2011
    Er = -0.02 #Table 1 of Robotham et al. 2011
    nu = 0.63 #Table 1 of Robotham et al. 2011
    delta = 9 #Section 3.2 of Robotham et al. 2011
    r_delta = (1.5/h).to(u.pc * u.s / u.km).value #Section 3.2 of Robotham et al. 2011
    l_delta = (12/h).to(u.pc * u.s / u.km).value #Section 3.2 of Robotham et al. 2011
    
    print('Constants used : H0 =', H0, ', Om0 =', Om0, ', Ode0 =', Ode0, ', h =', h, ', alpha =', alpha, ', M_star =', M_star, ', phi_star =', phi_star, ', b0 =', b0, ', R0 =', R0, ', Eb =', Eb, ', Er =', Er, ', nu =', nu, ', delta =', delta, ', r_delta =', r_delta, ', l_delta =', l_delta)
    
    print('Setting up the Group finder : ')
    
    dataframe = dataframe[(dataframe[Z_col]>zmin) & (dataframe[Z_col]<zmax)]
    dataframe['Lum_Distance'] = (cosmo.luminosity_distance(dataframe[Z_col]))*1e6
    dataframe['Co_Distance'] = cosmo.comoving_distance(dataframe[Z_col]).to(u.pc).value
    dataframe['AbMag'] = dataframe[mag_col] - 5 * np.log10(dataframe['Lum_Distance']) + 5 - dataframe[Kcorr_col]
    
    print('Calculating rho_bar ...')
    
    dataframe['Sch'] = 0.4 * np.log(10) * phi_star * (10**(0.4*(M_star - dataframe['AbMag'])*(alpha+1))) * np.exp(-10**(0.4*(M_star - dataframe['AbMag'])))
    
    low_int_lim = np.array(np.maximum((min(dataframe[mag_col]) - 5 * np.log10(dataframe['Lum_Distance']) + 5), min(dataframe['AbMag'])))
    upp_int_lim = np.array(np.minimum((max(dataframe[mag_col]) - 5 * np.log10(dataframe['Lum_Distance']) + 5), max(dataframe['AbMag'])))

    denom = scipy.integrate.quad(sch, min(dataframe['AbMag']), max(dataframe['AbMag']), args=(alpha, M_star, phi_star))[0]  
    
    Sz = []
    for low_lim, upp_lim in tqdm(zip(low_int_lim, upp_int_lim)):
        num = scipy.integrate.quad(sch, low_lim, upp_lim, args=(alpha, M_star, phi_star))[0]
        Sz.append(num / denom)

    dataframe['rho_bar'] = dataframe['Sch'] * Sz

    print('Calculating rho_emp ...')
    
    dataframe['Radius_link'] = r_delta/(dataframe['Co_Distance'])
    Radial_link = l_delta
    Radius=np.array(list(zip((dataframe[RA_col]).apply(math.radians),(dataframe[Dec_col]).apply(math.radians))))
    Radius_tree = KDTree(Radius, leaf_size=2) 
    Radial = np.array(list(zip(dataframe['Co_Distance'])))

    dataframe['IndexLoc'] = dataframe.reset_index().index
    Num_Den=[]
    for index in tqdm(dataframe.index):
        Radius_ind = Radius_tree.query_radius(Radius[[dataframe['IndexLoc'][index]]], r=dataframe['Radius_link'][index])[0]
        Radial_tree = KDTree(Radial[Radius_ind], leaf_size=2)
        Radial_ind=Radial_tree.query_radius(Radial[[dataframe['IndexLoc'][index]]], r=Radial_link)[0]
        Num_Den.append(len((dataframe.iloc[Radius_ind]).iloc[Radial_ind].index)) 
    dataframe['Num_Den']=Num_Den

    rho_emp_arr=[]
    for i in tqdm(dataframe.index):
        rho_emp_arr.append(dataframe['Num_Den'][i] / ((np.pi) * ((r_delta/1e6)**2) * (l_delta/1e6)))
    dataframe['rho_emp']=rho_emp_arr
    
    print('Calculating R and b ...')
    
    b_arr=[]    
    R_arr=[]
    for i in tqdm(dataframe.index):
        b_arr.append(b0*(dataframe['rho_emp'][i]/(dataframe['rho_bar'][i] * delta))**Eb) # equation 5 of Robotham et al. 2011
        R_arr.append(R0*(dataframe['rho_emp'][i]/(dataframe['rho_bar'][i] * delta))**Er) # equation 6 of Robotham et al. 2011
    dataframe['b']=b_arr
    dataframe['R']=R_arr
    
    print('Calculating D_lim ...')
    
    term_1 = np.array(sch(max(dataframe[mag_col]) - 5 * np.log10(dataframe['Lum_Distance']) + 5, alpha, M_star, phi_star) / dataframe['Sch'])    
    
    upp_int_lim = np.array(max(dataframe[mag_col]) - 5 * np.log10(dataframe['Lum_Distance']) + 5)

    D0 = []
    for i in tqdm(range(len(upp_int_lim))):
        term_2 = scipy.integrate.quad(sch, -100, upp_int_lim[i], args=(alpha, M_star, phi_star))[0]
        D0.append(1e6 * (term_1[i])**(nu/3) * (term_2)**(-1/3)) # equation 3 of Robotham et al. 2011

    dataframe['D0']=D0
    
    prev_len = len(dataframe)
    dataframe=dataframe[(dataframe['D0']<10**(8))]
    new_len = len(dataframe)
    print(f"{prev_len - new_len} galaxies removed due to having D_lim > 10**8")
    
    dataframe['Ang_link']=np.arctan(dataframe['b']*dataframe['D0']/(dataframe['Co_Distance']))
    dataframe['Los_link']=dataframe['b']*dataframe['R']*dataframe['D0']
    
    ANG=np.array(list(zip((dataframe[RA_col]).apply(math.radians),(dataframe[Dec_col]).apply(math.radians))))
    ANGtree = KDTree(ANG, leaf_size=2) 
    
    D=np.array(list(zip(dataframe['Co_Distance'])))
    
    dataframe['IndexLoc'] = dataframe.reset_index().index
    
    print('Running the Group finder : ')
    
    print('Finding groups ...')
    
    groups={}
    galgroupdict={}

    drop=[]
    centre=[]

    for ix in tqdm(dataframe.index):
        if ix not in drop and ix not in centre:
            centre.append(ix)
            groups[ix] = []
            j=0
            while True:
                try: 
                    if j == 0:
                        index = ix
                    else:
                        index = groups[ix][j]
                    ind = ANGtree.query_radius(ANG[[dataframe['IndexLoc'][index]]], r=dataframe['Ang_link'][index])[0]
                    Dtree = KDTree(D[ind], leaf_size=2)
                    Dind=Dtree.query_radius(D[[dataframe['IndexLoc'][index]]], r=dataframe['Los_link'][index])[0]
                    for i in (dataframe.iloc[ind]).iloc[Dind].index:
                        if i not in drop:
                            groups[ix].append(i)
                            drop.append(i)
                            galgroupdict[i]=ix
                    j+=1
                except IndexError:
                    break

    for i in groups:
        if len(groups[i]) == 1:
            galgroupdict[i] = 0

    grouplist = []
    for i in dataframe.index:
        grouplist.append(galgroupdict[i])
    dataframe['RecID'] = grouplist    
    
    print('Writing groups to a dataframe ...')
    
    dfRec, recdict=matchGAMA.start(dataframe,'RecID')
    
    RecCentres = []

    for group in tqdm(dfRec.index):
        gals=recdict[group].copy()

        while len(gals)>2:
            RA = []
            Dec = []
            Lum = []
            distlumcen = {}
            for galaxy in gals:
                RA.append(dataframe[RA_col][galaxy])
                Dec.append(dataframe[Dec_col][galaxy])
                Lum.append(10**(-0.4*dataframe['AbMag'][galaxy]))

            lumcenRA = np.average(RA,weights=Lum)
            lumcenDec = np.average(Dec,weights=Lum)

            for galaxy in gals:
                distlumcen[galaxy]=(abs(math.sqrt(((dataframe[RA_col][galaxy]-lumcenRA)**2)+((dataframe[Dec_col][galaxy]-lumcenDec)**2))))

            gals.remove(max(distlumcen,key=distlumcen.get))

        mags={}
        mags[gals[0]]=10**(-0.4*dataframe['AbMag'][gals[0]])
        mags[gals[1]]=10**(-0.4*dataframe['AbMag'][gals[1]])
        RecCentres.append(max(mags, key=mags.get))

    dfRec['CentreID'] = RecCentres

    #Defining the Z, RA and Dec of each galaxy group
    RecCentreZ =[]
    RecCentreRA = []
    RecCentreDec = []
    RecCentreCATAID = []
    for galaxy in dfRec['CentreID']:
        RecCentreZ.append(dataframe[Z_col][galaxy])
        RecCentreRA.append(dataframe[RA_col][galaxy])
        RecCentreDec.append(dataframe[Dec_col][galaxy])
        RecCentreCATAID.append(dataframe[ID_col][galaxy])

    dfRec['Z'] = RecCentreZ
    dfRec['RA'] = RecCentreRA
    dfRec['Dec'] = RecCentreDec
    dfRec['CentreCATAID'] = RecCentreCATAID
    dfRec = dfRec.reset_index(drop=False)

    print('Calculating group velocity dispersion ...')

    dataframe['v'] = dataframe[Z_col] * c.value
    groupdictvel={}
    for i in tqdm(dfRec['RecID']):
        groupdictvel[i] = []
        for j in list(dataframe[dataframe['RecID']==i].index):
            groupdictvel[i].append(dataframe['v'][j])

    for i in groupdictvel:
        groupdictvel[i] = sorted(groupdictvel[i])

    sraw = []
    s = []
    for i in dfRec['RecID']:
        N = len(groupdictvel[i])
        wgsum=0
        sumerr=0
        for j in range(1,N):
            w = (j * (N-j))
            g = (groupdictvel[i][j] - groupdictvel[i][j-1])/1e3
            wgsum += w * g / (1+dfRec['Z'][np.array(dfRec[dfRec['RecID']==i].index.values)[0]])
        for k in list(dataframe[dataframe['RecID']==i].index):
            err=dataframe[vel_err_col][k]
            sumerr += (err**2)
        sgap = (np.sqrt(np.pi)/(N*(N-1))) * wgsum # equation 16 of Robotham et al. 2011
        serr = np.sqrt(sumerr/N)
        sraw.append(sgap * np.sqrt(N/(N-1)))
        if serr > sgap:
            s.append(0)
        else:
            s.append(np.sqrt(((N/(N-1)) * sgap**2) - serr**2)) # equation 17 of Robotham et al. 2011

    dfRec['VelDisp_raw'] = np.array(sraw)
    dfRec['VelDisp'] = np.array(s)

    prev_len = len(dfRec)
    dfRec=dfRec[(dfRec['VelDisp']<2500)]
    new_len = len(dfRec)
    print(f"{prev_len - new_len} groups removed due to having Vel_Disp > 2500")

    print('Calculating group projected radius ...')
    
    angdist = {k: [] for k in dfRec['RecID']}
    for group in tqdm(dfRec['RecID']):
        gals=recdict[group]
        for gal in gals:
            angdist[group].append(np.array(dataframe['Co_Distance'][gal] * np.arcsin((np.pi/180) * pyasl.getAngDist(dfRec['RA'][dfRec[dfRec['RecID']==group].index.values], dfRec['Dec'][dfRec[dfRec['RecID']==group].index.values], dataframe[RA_col][gal], dataframe[Dec_col][gal])))[0])

    rad = []
    for i in tqdm(dfRec['RecID']):
        if np.array(dfRec['N'][dfRec[dfRec['RecID']==i].index.values])==2:
            rad.append(angdist[i][0]/2)
        else :
            rad.append(np.percentile(angdist[i],50))
    dfRec['Rad']=np.array(rad)/1e6      
    
    print('Calculating group dynamical mass ...')

    DynMass=[]
    for i in tqdm(dfRec.index) :
        DynMass.append((10/(4.301*10**(-9))) * (dfRec['VelDisp'][i])**2 * dfRec['Rad'][i]) # equation 18 of Robotham et al. 2011
    dfRec['DynMass']=DynMass  
    
    dfRec = dfRec.reset_index(drop=True)
    dataframe = dataframe.reset_index(drop=True)

    Dec_Rad = dataframe[Dec_col].apply(math.radians)
    RA_Rad = dataframe[RA_col].apply(math.radians)
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, polar=True)
    ax.scatter(RA_Rad, dataframe[Z_col], s=5, label='Galaxies', color='blue')
    group_sizes = 20 * dfRec['N']
    ax.scatter(dfRec['RA'].apply(math.radians), dfRec['Z'], s=group_sizes, alpha=0.5, label='Groups', color='red')
    ax.set_thetamin(math.degrees(min(RA_Rad)))
    ax.set_thetamax(math.degrees(max(RA_Rad)))
    ax.set_rlabel_position(-22.5)
    ax.grid(True)
    plt.title(f'{zmin} < z < {zmax}')
    plt.legend(loc='upper right', shadow=True, fontsize=12)
    plt.show()
    
    return dfRec, dataframe
    
    
   
def sch(M, alpha, M_star, phi_star):
    """Schechter function, defined in equation 4 of Loveday et al. 2012"""

    return 0.4 * np.log(10) * phi_star * (10**(0.4*(M_star - M)*(alpha+1))) * np.exp(-10**(0.4*(M_star - M)))

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.constants import c, G
from astropy.cosmology import LambdaCDM
import scipy
from tqdm.notebook import tqdm
import pandas as pd
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Main procedures
#------------------------------------------------------------------------------

def find_groups(dataframe, mlims=(0, 19.65), Mlims=(-24, -12), m_col='m_r', M_col='M_r', z_col='Z', ra_col='RA', dec_col='DEC', 
                velerr_col='VEL_ERR', z_comp_col='z_comp', galid_col = 'uberID', 
                H0=100, Om0=0.25, Ode0=0.75, M_star=-20.71, alpha=-1.26, phi=10**(-2.02), 
                delta=9, r_delta=1.5, l_delta=12, b0=0.06, R0=18, E_b=-0.00, E_r=-0.02, nu=0.63, A=10.0, B=1.04):
    """Performs group finding and calculates group velocity dispersion, projected radius, dynamical mass and luminosity"""

    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    filtered_dataframe = dataframe[(dataframe[m_col]>mlims[0]) & (dataframe[m_col]<mlims[1]) & 
                                   (dataframe[M_col]>Mlims[0]) & (dataframe[M_col]<Mlims[1])]
    print(len(dataframe)-len(filtered_dataframe), 'galaxies filtered out')
    print('Processing', len(filtered_dataframe), 'galaxies ...')

    ra = filtered_dataframe[ra_col].values
    dec = filtered_dataframe[dec_col].values
    z = filtered_dataframe[z_col].values
    v_err = filtered_dataframe[velerr_col].values
    m_vals = filtered_dataframe[m_col].values
    M_vals = filtered_dataframe[M_col].values
    z_comp_vals = filtered_dataframe[z_comp_col].values
    galid_vals = filtered_dataframe[galid_col].values

    coords = np.radians(np.column_stack((ra, dec))) #rad
    num_galaxies = len(filtered_dataframe)
    ang_kdtree = cKDTree(coords)

    print('Calculating mean comoving galaxy separation and comoving radial distance ...')
    lum_distances = lum_dist(z, cosmo) #pc
    Mmax_values = calc_Mlims(lum_distances, mlims)[1]

    sch_values = sch(M_vals, M_star, alpha, phi) #Mpc^-3
    sch_Mmax_values = sch(Mmax_values, M_star, alpha, phi) #Mpc^-3
    sch_integral_values = np.array([scipy.integrate.quad(sch, -100, Mmax, args=(M_star, alpha, phi))[0] for Mmax in tqdm(Mmax_values)]) #Mpc^-3

    D_lim = (sch_Mmax_values / sch_values) ** (nu/3) * (sch_integral_values ** (-1/3)) #Robotham et al. 2011 equation 3, Mpc
    print(np.sum(D_lim>100), 'galaxies had their D_lim value clipped at 100 Mpc')
    D_lim = np.clip(D_lim, None, 100)  # Cap upper bound at 100 Mpc
    D_com = co_dist(z, cosmo) #Mpc

    print('Calculating mean required linking overdensity and radial expansion factor ...')
    rho_bar_values = sch_values * Sz(z, mlims, Mlims, M_star, alpha, phi, cosmo) #Robotham et al. 2011 section 2.1.3, Mpc^-3
    rho_emp_values = rho_emp(ang_kdtree, ra, dec, D_com, r_delta, l_delta) #Robotham et al. 2011 section 2.1.3, Mpc^-3
    b = b0 * (((1/delta) * (rho_emp_values/rho_bar_values))**(E_b)) #Robotham et al. 2011 equation 5, unitless
    R = R0 * (((1/delta) * (rho_emp_values/rho_bar_values))**(E_r)) #Robotham et al. 2011 equation 6, unitless
    b_comp = b / (z_comp_vals**(1/3)) #Robotham et al. 2011 equation 7 
    ang_search=np.arctan(b_comp*D_lim/D_com) #maximum value of Robotham et al. 2011 equation 1, rad
    los_search=b_comp*R*D_lim #maximum value of Robotham et al. 2011 equation 4, Mpc

    # Re-order galaxies by descending search linking lengths to avoid having galaxies belonging to a large group first form into a small group
    sorted_indices = np.lexsort((los_search, ang_search))[::-1]
    filtered_dataframe['old_index'] = filtered_dataframe.index.values
    filtered_dataframe = filtered_dataframe.iloc[sorted_indices].reset_index(drop=True)
    ra = ra[sorted_indices]
    dec = dec[sorted_indices]
    z = z[sorted_indices]
    v_err = v_err[sorted_indices]
    m_vals = m_vals[sorted_indices]
    M_vals = M_vals[sorted_indices]
    coords = coords[sorted_indices]
    z_comp_vals = z_comp_vals[sorted_indices]
    galid_vals = galid_vals[sorted_indices]
    ang_kdtree = cKDTree(coords)
    lum_distances = lum_distances[sorted_indices]
    Mmax_values = Mmax_values[sorted_indices]
    sch_values = sch_values[sorted_indices]
    sch_Mmax_values = sch_Mmax_values[sorted_indices]
    sch_integral_values = sch_integral_values[sorted_indices]
    D_lim = D_lim[sorted_indices]
    D_com = D_com[sorted_indices]
    rho_bar_values = rho_bar_values[sorted_indices]
    rho_emp_values = rho_emp_values[sorted_indices]
    b = b[sorted_indices]
    R = R[sorted_indices]
    b_comp = b_comp[sorted_indices]
    ang_search = ang_search[sorted_indices]
    los_search = los_search[sorted_indices]
    v = z * c.to(u.km / u.s).value #km.s^-1

    print('Finding groups ...')
    visited = np.zeros(num_galaxies, dtype=bool)
    groups = []
    groups_cen = []
    groups_cen_id = []
    groups_len = []
    groups_ra = []
    groups_dec = []
    groups_z = []
    groups_vel_disp = []
    groups_proj_rad = []
    groups_dyn_mass = []
    groups_lum = []
    groups_members = []
    groups_members_id = []
    for i in tqdm(range(num_galaxies)):
        if visited[i]:
            continue

        group = set()
        stack = [i]

        while stack:
            current = stack.pop()
            if visited[current]:
                continue

            visited[current] = True
            group.add(current)

            # Compute comoving parameters
            ang_search_current = ang_search[current]
            los_search_current = los_search[current]
            b_comp_current = b_comp[current]
            R_current = R[current]
            D_com_1, D_lim_1 = D_com[current], D_lim[current]

            # Find projected neighbors
            projected_neighbors = ang_kdtree.query_ball_point(coords[current], r=ang_search_current)  # Max angle search radius
            group_rad_kdtree = cKDTree(D_com[projected_neighbors].reshape(-1, 1))
            radial_neighbors_indices = group_rad_kdtree.query_ball_point(D_com[current], r=los_search_current)  # Max los search radius
            radial_neighbors = [projected_neighbors[idx] for idx in radial_neighbors_indices]
            common_neighbors = set(projected_neighbors).intersection(radial_neighbors)

            if len(common_neighbors) > 1:
                for neighbor in common_neighbors:
                    if neighbor == current or visited[neighbor]:
                        continue

                    D_com_2, D_lim_2 = D_com[neighbor], D_lim[neighbor]

                    # Compute angular separation
                    theta = angular_separation(ra[current], dec[current], 
                                            ra[neighbor], dec[neighbor])

                    # Compute mean distances
                    D_mean = (D_com_1 + D_com_2) / 2
                    D_lim_mean = (D_lim_1 + D_lim_2) / 2

                    # Projection condition
                    tan_theta = np.tan(theta)
                    proj_criterion = tan_theta <= b_comp_current * D_lim_mean / D_mean #Robotham et al. 2011 equation 1

                    # Radial condition
                    radial_criterion = abs(D_com_1 - D_com_2) <= b_comp_current * R_current * D_lim_mean #Robotham et al. 2011 equation 4

                    # Add to stack if both conditions met
                    if proj_criterion and radial_criterion:
                        stack.append(neighbor)

            # for neighbor in common_neighbors:
            #     if neighbor == current or visited[neighbor]:
            #         continue

            #     # Add to stack if not visited
            #     stack.append(neighbor)

        if len(group)>1:
            groups.append(group)
            groups_len.append(len(group))
            groups_members.append(group)
            groups_members_id.append([galid_vals[gal] for gal in group])

            # Searching for group central galaxy following description given for iterative center in Robotham et al. 2011 section 4.2.1
            group_gals = np.array(list(group), dtype=int)
            ras = ra[group_gals]
            decs = dec[group_gals]
            weights = 10**(-0.4*m_vals[group_gals])
            while len(group_gals) > 2:
                lumcen_ra = np.average(ras, weights=weights)
                lumcen_dec = np.average(decs, weights=weights)
                dist_lumcen_sq = (ras - lumcen_ra)**2 + (decs - lumcen_dec)**2
                worst_idx = np.argmax(dist_lumcen_sq)
                group_gals = np.delete(group_gals, worst_idx)
                ras = np.delete(ras, worst_idx)
                decs = np.delete(decs, worst_idx)
                weights = np.delete(weights, worst_idx)
            mags = m_vals[group_gals]
            groupcen = group_gals[np.argmin(mags)]

            # Calculating group velocity dispersion, projected radius, dynamical mass and luminosity following Robotham et al. 2011 sections 4.1, 4.2.3, 4.3 and 4.4
            group_gals = np.array(list(group), dtype=int)
            v_gals = v[group_gals]
            v_gals_err = v_err[group_gals]
            vel_disp = velocity_dispersion(v_gals, v_gals_err, z[groupcen])
            ra_gals = ra[group_gals]
            dec_gals = dec[group_gals]
            dist_gals = D_com[group_gals]
            proj_rad = group_radius(ra_gals, dec_gals, ra[groupcen], dec[groupcen], dist_gals)
            dyn_mass = dynamical_mass(vel_disp, proj_rad, A)
            lum = group_luminosity(M_vals[group_gals], z[groupcen], cosmo, mlims, M_star, alpha, phi, B)

            groups_cen.append(groupcen)
            groups_cen_id.append(galid_vals[groupcen])
            groups_ra.append(ra[groupcen])
            groups_dec.append(dec[groupcen])
            groups_z.append(z[groupcen])
            groups_vel_disp.append(vel_disp)
            groups_proj_rad.append(proj_rad)
            groups_dyn_mass.append(dyn_mass)
            groups_lum.append(lum)

    filtered_dataframe['Lum_Dist'] = lum_distances
    filtered_dataframe['Sch'] = sch_values
    filtered_dataframe['D_lim'] = D_lim
    filtered_dataframe['D_com'] = D_com
    filtered_dataframe['rho_bar'] = rho_bar_values
    filtered_dataframe['rho_emp'] = rho_emp_values
    filtered_dataframe['b'] = b
    filtered_dataframe['R'] = R
    filtered_dataframe['b_comp'] = b_comp
    filtered_dataframe['anglink'] = ang_search
    filtered_dataframe['loslink'] = los_search

    group_ids = list(range(1, len(groups) + 1))
    group_dataframe = pd.DataFrame(data = {'GroupID': group_ids, 'N': groups_len, 'GalCen': groups_cen, 'RA': groups_ra, 'Dec': groups_dec, 'z': groups_z, 
                                           'vel_disp': groups_vel_disp, 'rad50': groups_proj_rad, 'dyn_mass': groups_dyn_mass, 'lum': groups_lum, 
                                           'Gals': groups_members, 'GalCenID': groups_cen_id, 'Gals_id': groups_members_id})

    group_assignments = np.zeros(len(filtered_dataframe), dtype=int)
    for group_id, member_indices in zip(group_ids, groups):
        group_assignments[list(member_indices)] = group_id
    filtered_dataframe['GroupID'] = group_assignments


    return filtered_dataframe, group_dataframe

def calc_z_comp(dataframe, search_radius=1.0, NQ_col='NQ', z_col='Z', ra_col='RA', dec_col='DEC', H0=100, Om0=0.3, Ode0=0.7):
    """Calculates the local redshift completeness of a given galaxy dataset"""

    cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    dataframe = dataframe[dataframe[z_col]>0]
    ra = dataframe[ra_col].values
    dec = dataframe[dec_col].values
    z = dataframe[z_col].values
    NQ = dataframe[NQ_col].values
    coords = np.radians(np.column_stack((ra, dec))) #rad
    ang_kdtree = cKDTree(coords)
    D_com = co_dist(z, cosmo) #Mpc

    completeness = np.ones(len(NQ))
    for i in tqdm(range(len(NQ)), desc="Calculating completeness"):
        neighbors = ang_kdtree.query_ball_point(coords[i], r=search_radius / D_com[i])
        N_total = len(neighbors)
        N_reliable = np.sum(NQ[neighbors] > 2)
        completeness[i] = N_reliable / N_total if N_total > 0 else 1.0 #fraction of neighbouring galaxies with reliable redhsifts or 1.0 if no neighbours
    dataframe['z_comp'] = np.clip(completeness, 0, 1)

    return dataframe

def plot_galaxy_groups(galaxy_dataframe, group_dataframe, gal_ra_col='RAcen', gal_z_col='Z', group_ra_col='RA' , group_z_col='z', group_N_col='N', zmin=0.002, zmax=0.65):
    """Makes a polar plot of the galaxies and galaxy groups in the datasets provided"""

    galaxy_filtered = galaxy_dataframe[(galaxy_dataframe[gal_z_col] >= zmin) & (galaxy_dataframe[gal_z_col] <= zmax)]
    group_filtered = group_dataframe[(group_dataframe[group_z_col] >= zmin) & (group_dataframe[group_z_col] <= zmax)]

    RA_rad = np.radians(galaxy_filtered[gal_ra_col].values)
    Z = galaxy_filtered[gal_z_col]

    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot(111, polar=True)

    #galaxy plot
    ax.scatter(RA_rad, Z, s=5, label='Galaxies', color='blue')

    #group plot with size proportional to 20*N
    group_sizes = 20 * group_filtered[group_N_col]
    group_RA_rad = np.radians(group_filtered[group_ra_col].values)
    group_Z = group_filtered[group_z_col]
    ax.scatter(group_RA_rad, group_Z, s=group_sizes, alpha=0.5, label='Groups', color='red', edgecolors='black')

    if len(RA_rad) > 0:
        ax.set_thetamin(np.degrees(RA_rad.min()))
        ax.set_thetamax(np.degrees(RA_rad.max()))
    ax.set_rlabel_position(-22.5)
    ax.grid(True)

    plt.title(f'Galaxy Groups : {zmin} < z < {zmax}', fontsize=18)
    plt.legend(loc='upper right', shadow=True, fontsize=12)
    plt.show()

#------------------------------------------------------------------------------
# Support functions
#------------------------------------------------------------------------------

def calc_Mlims(distance, mlims):
    """Calculates the effective absolute magnitude limits for given apparent magnitude limits at a given distance"""
    return mlims[0] - 5*np.log10(distance) + 5, mlims[1] - 5*np.log10(distance) + 5

def sch(M, M_star, alpha, phi):
    """Calculates Schechter function evaluation as defined in equation 4 of Loveday et al. 2012"""
    return 0.4 * np.log(10) * phi * (10**(0.4*(M_star - M)*(alpha+1))) * np.exp(-10**(0.4*(M_star - M)))

def lum_dist(z, cosmo):
    """Calculates the luminosity distance for a given redshift in pc"""
    return cosmo.luminosity_distance(z).to(u.parsec).value

def angular_separation(ra1, dec1, ra2, dec2):
    """Calculates the angular separation between two points on the sky"""
    # coord1 = SkyCoord(ra=ra1, dec=dec1, unit=u.deg)
    # coord2 = SkyCoord(ra=ra2, dec=dec2, unit=u.deg)
    # return coord1.separation(coord2).radian

    ra1, dec1 = np.radians(ra1), np.radians(dec1)
    ra2, dec2 = np.radians(ra2), np.radians(dec2)
    sin_d1, sin_d2 = np.sin(dec1), np.sin(dec2)
    cos_d1, cos_d2 = np.cos(dec1), np.cos(dec2)
    delta_ra = ra2 - ra1
    cos_delta_ra = np.cos(delta_ra)
    cos_angle = sin_d1 * sin_d2 + cos_d1 * cos_d2 * cos_delta_ra
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))

def co_dist(z, cosmo):
    """Calculates the comoving distance for a given redshift in Mpc"""
    return cosmo.comoving_distance(z).value

def rho_emp(ang_kdtree, ra_gals, dec_gals, co_dist_gals, r_delta, l_delta):
    """Calculates the empirically estimated density as defined in section 2.1.3 Robotham et al. 2011"""
    coords = np.radians(np.column_stack((ra_gals, dec_gals)))
    proj_neighbors = ang_kdtree.query_ball_point(coords, r=r_delta/co_dist_gals)
    rho_emp = []
    for i in tqdm(range(len(ra_gals))):
        n_gals = np.sum(np.abs(co_dist_gals[i] - co_dist_gals[proj_neighbors[i]]) <= l_delta)
        rho_emp.append(n_gals / (np.pi * r_delta**2 * l_delta))
    return np.array(rho_emp)

def Sz(z, mlims, Mlims, M_star, alpha, phi, cosmo): 
    """Calculates selection function as defined in equation 11 of Loveday et al. 2012"""
    Mlo, Mhi = Mlims
    Mmin, Mmax = calc_Mlims(lum_dist(z, cosmo), mlims)
    lower_bound = np.maximum(Mmin, Mlo)
    upper_bound = np.minimum(Mmax, Mhi)
    Sz = []
    for low_bound, up_bound in tqdm(zip(lower_bound, upper_bound)):
        Sz.append(scipy.integrate.quad(sch, low_bound, up_bound, args=(M_star, alpha, phi))[0] / 
                  scipy.integrate.quad(sch, Mlo, Mhi, args=(M_star, alpha, phi))[0]) 
    return Sz

def velocity_dispersion(v_gals, v_gals_err, z_group):
    """Calculates group velocity dispersion as defined in section 4.1 of Robotham et al. 2011"""
    N = len(v_gals)
    v_sorted = np.sort(v_gals)
    gaps = (v_sorted[1:] - v_sorted[:-1]) / (1+z_group)
    weights = np.array([i * (N - i) for i in range(1, N)])
    sgap = (np.sqrt(np.pi)/(N*(N-1))) * np.sum(weights * gaps)
    serr = np.sqrt(np.sum(v_gals_err**2)/N)
    vel_disp = np.sqrt(max(0, ((N/(N-1)) * sgap**2) - serr**2))
    return vel_disp
    
def group_radius(ra_gals, dec_gals, ra_cen_gal, dec_cen_gal, dist_gals):
    """Calculates group rad50 projected radius as defined in section 4.2.3 of Robotham et al. 2011"""
    angular_seps = angular_separation(ra_cen_gal, dec_cen_gal, ra_gals, dec_gals)
    physical_seps = angular_seps * dist_gals
    sorted_seps = np.sort(physical_seps)
    rad_50 = np.percentile(sorted_seps, 50)
    return rad_50

def dynamical_mass(vel_disp, proj_rad, A):
    """Calculates group dynamical mass as defined in section 4.3 of Robotham et al. 2011"""
    return (A / G.to(u.Mpc * (u.km)**2 / u.M_sun / (u.s)**2).value) * vel_disp**2 * proj_rad 

def group_luminosity(M_gals, z_group, cosmo, mlims, M_star, alpha, phi, B):
    """Calculates group luminosity as defined in section 4.4 of Robotham et al. 2011"""
    L_ob = np.sum(10**(0.4*(4.67-M_gals)))
    Mmax = calc_Mlims(lum_dist(z_group, cosmo), mlims)[1]
    integrand = lambda M: 10**(-0.4 * M) * sch(M, M_star, alpha, phi)
    frac = (scipy.integrate.quad(integrand, -30, -14)[0] / scipy.integrate.quad(integrand, -30, Mmax)[0])
    return B * L_ob * frac 

function read_element(energy_center, energies=Sif1f2)
    for i in 1:size(energies)[1]
        if energies[max(1, i - 1), 1] < energy_center <= energies[i, 1]
            return energies[i, 2], energies[i, 3]
        end
    end

    error("No energy found")
end

function structure_factor(hkl, energy_center; element = :Si,
                             a_Par=5.431, b_Par=5.431, c_Par=5.431)
    if element == :C

        Z = 6
        
        #Cell parameters
        a_Par = 3.567
        b_Par = 3.567
        c_Par = 3.567
        #Cell angles
        α_par = 90
        β_par = 90
        γ_par = 90
        
        #cell_structure = 'dia' #Diamond cell strucuture

        #Reflections and atomic structure factor
        f_0 = if hkl in ((4,0,0), (0,4,0), (0,0,4))
            1.6077047        
        elseif hkl == (1,1,1)
            3.1379116            
        elseif hkl in ((2,2,0), (2,0,2), (0,2,2))
            2.029862            
        elseif hkl in ((3,1,1), (1,3,1), (1,1,3))
            1.7972306            
        elseif hkl in ((3,1,3), (3,3,1), (1,3,3))
            1.5400894           
        elseif hkl in ((5,1,1), (1,5,1), (1,1,5))
            1.4097152        
        elseif hkl in ((2,2,4), (4,2,2), (2,4,2))
            1.454006        
        elseif hkl == (4,4,4)
            1.1557
        elseif hkl in ((5,3,3), (3,5,3), (3,3,5))
            1.20988
        elseif hkl in ((5,1,3), (5,3,1), (3,5,1),(3,1,5),(1,5,3),(1,3,5))
            1.3040249
        
        elseif hkl in ((4,4,0), (4,0,4),(0,4,4))
            1.3420491
        elseif hkl in ((5,1,5),(1,5,5),(5,5,1))
            1.1248178
        elseif hkl == (3,3,3)
            1.4097152
        else
            error("Unsupported Miller indices: $(hkl)")
        end

        # Imaginary and real part of the structure factor from table https://henke.lbl.gov/optical_constants/asf.html
        # num = xlsread('Diamond_f1_f2.xlsx');
        #num = xlsread('elements/Cf1f2.xlsx'); #Matlab
        f_1, f_2 = read_element(energy_center, Cf1f2)
        h, k, l = hkl
         
        coef = (1 + (-1)^(h + k) + (-1)^(h + l) + (-1)^(k + l)) * (1 + (-1im)^(h + k + l))
        
        F0 = (Z + f_1 + 1im * f_2) * 8
        FH = (f_0 + f_1 + 1im * f_2) * abs(coef)
        F_H = (f_0 + f_1 + 1im * f_2) * abs(coef)
        
    elseif element == :Si
        Z = 14
        
        a_Par=5.431
        b_Par=5.431
        c_Par=5.431
        
        α_par = 90
        β_par = 90
        γ_par = 90
        
        f_0 = if hkl in ((5, 1, 3), (5,3,1), (3,5,1), (3,1,5), (1,5,3), (1,3,5))
            5.92227
        elseif hkl in ((3,1,1), (1,3,1), (1,1,3))
            8.2626997
        elseif hkl == (1,1,1)
            10.702846
        elseif hkl in ((2,2,0), (2,0,2), (0,2,2))
            8.8399534
        elseif hkl in ((4,0,0), (0,4,0), (0,0,4))
            7.5887902
        elseif hkl in ((3,3,1), (1,3,3), (3,1,3))
            7.2654482
        elseif hkl in ((4,4,0), (0,4,4), (4,0,4))
            6.143148
        elseif hkl == (3,3,3)
            6.5355327
        elseif hkl == (5,5,5)
            3.8871919
        elseif hkl in ((1,5,5), (5,1,5), (5,5,1))
            4.9354269
        elseif hkl in ((5,5,3), (5,3,5), (3,5,5))
            4.5371446
        elseif hkl in ((3,3,5), (3,5,3), (5,3,3))
            5.3936593
        elseif hkl in ((8,0,0), (0,8,0), (0,0,8))
            4.3147635
        elseif hkl in ((0,4,8), (4,0,8), (8,4,8), (8,4,0), (4,8,0), (0,8,4))
            3.585777
        else
            error("Unsupported Miller indices: $(hkl)")
        end
        
        f_1, f_2 = read_element(energy_center, Sif1f2)
        h, k, l = hkl
        
        coef = (1 + (-1)^(h + k) + (-1)^(h + l) + (-1)^(k + l)) * (1 + (-1im)^(h + k + l))
        F0 = (Z + f_1 + 1im * f_2) * 8
        FH = (f_0 + f_1 + 1im * f_2) * abs(coef)
        F_H = (f_0 + f_1 + 1im * f_2) * abs(coef)
        
    elseif element == :Ge
        
        Z = 32
        
        a_Par = 5.658
        b_Par = 5.658
        c_Par = 5.658

        α_par = 90
        β_par = 90
        γ_par = 90
                
        f_0 = if hkl in ((5,1,3), (5,3,1), (3,5,1),(3,1,5),(1,5,3),(1,3,5))
            15.622178200533803                
        elseif hkl in ((3,1,1),(1,3,1),(1,1,3))
            22.378056270949067                    
        elseif hkl in ((1,1,1))
            27.367058878665116
        elseif hkl in ((2,2,0),(2,0,2),(0,2,2))
            23.797789346546946
        elseif hkl in ((4,0,0), (0,4,0), (0,0,4))
            20.461697153045826
        elseif hkl in ((3,3,1), (1,3,3), (3,1,3))
            19.486433565396258
        elseif hkl in ((4,4,0), (0,4,4), (4,0,4))
            16.21692019691965
        elseif hkl in ((3,3,3))
            17.324324213026493
        elseif hkl in ((5,5,5))
            10.665062488265056
        elseif hkl in ((1,5,5), (5,1,5), (5,5,1))
            13.103721306960553
        elseif hkl in ((5,5,3), (5,3,5), (3,5,5))
            12.150938489418056
        elseif hkl in ((3,3,5), (3,5,3), (5,3,3))
            14.242875380623753
        elseif hkl in ((8,0,0), (0,8,0), (0,0,8))
            11.63309041116516
        elseif hkl in ((4,4,4))
            13.506653614349618
        else
            error("Unsupported Miller indices: $(hkl)")
        end

        f_1, f_2 = read_element(energy_center, Gef1f2)
        h, k, l = hkl
        
        coef = (1 + (-1)^(h + k) + (-1)^(h + l) + (-1)^(k + l)) * (1 + (-1im)^(h + k + l))
        F0 = (Z + f_1 + 1im * f_2) * 8
        FH = (f_0 + f_1 + 1im * f_2) * abs(coef)
        F_H = (f_0 + f_1 + 1im * f_2) * abs(coef)
    end
    return (; Z, F0, FH, F_H, a_Par, b_Par, c_Par, α_par, β_par, γ_par)
end

"""
Definition of the strain profile.

- Thickness in micrometers um
- Step_layer is given in angstrom
- Step of the strain net in angstrom
"""
function Bipolar_with_surface_exp(thickness; n_layers=3, #step_layer=100_000_0,
        Strain_Val_a = 0.0, Strain_Val_b = 0.000, Strain_Val_c = 0.000,
        Interfase_Val_a = 10_000_0, Interfase_Val_b = 10_000_0, Interfase_Val_c = 10_000_0,
        cdepth_a = 500, cdepth_b = 500, cdepth_c = 500, Compressed_strain1 = false,
        Strain_Val_a2 = 0.0, Strain_Val_b2 = 0.0, Strain_Val_c2 = 0.0,
        Interfase_Val_a2 = 10_000_0, Interfase_Val_b2 = 10_000_0, Interfase_Val_c2 = 10_000_0,
        cdepth_a2 = 500, cdepth_b2 = 500, cdepth_c2 = 500, Compressed_strain2 = false)
    thickness_layer = thickness * 10^4  # Thickness to Angstrom
    step_layer = thickness_layer / n_layers

    #First wave
    #Strain_Val_a = 0.0         # Strain in the surface
    #Strain_Val_b = 0.0         # Strain in the surface
    #Strain_Val_c = 0.0         # Strain in the surface
    #Interfase_Val_a = 10_000_0  # Interdace Angstrom
    #Interfase_Val_b = 10_000_0  # Interdace Angstrom
    #Interfase_Val_c = 10_000_0  # Interdace Angstrom
    #cdepth_a = 500             # Center of the strain effect in a
    #cdepth_b = 500             # Center of the strain effect in b
    #cdepth_c = 500             # Center of the strain effect in c

    #Compressed_strain1 = false # Compressive True and Expansive False


    #Second Wave
    #Strain_Val_a2 = 0.0         # Strain in the surface
    #Strain_Val_b2 = 0.0         # Strain in the surface
    #Strain_Val_c2 = 0.0         # Strain in the surface
    #Interfase_Val_a2 = 10_000_0 # Interdace Angstrom
    #Interfase_Val_b2 = 10_000_0 # Interdace Angstrom
    #Interfase_Val_c2 = 10_000_0 # Interdace Angstrom
    #cdepth_a2 = 500             # Center of the strain effect in a
    #cdepth_b2 = 500             # Center of the strain effect in b
    #cdepth_c2 = 500             # Center of the strain effect in c

    #Compressed_strain2 = false  # Compressive True and Expansive False

    n_steps_layers = round(Int, thickness_layer / step_layer) #Calculation of the number of layers

    # generation of the net
    x_ISD = if n_steps_layers == 1
        [thickness_layer]
    else
        range(1, thickness_layer, n_steps_layers)
    end


    #Front Surface
    R_coefficient_exp_a = 1/(10*Interfase_Val_a)
    R_coefficient_exp_b = 1/(10*Interfase_Val_b)
    R_coefficient_exp_c = 1/(10*Interfase_Val_c)

    #Rear surface
    R_coefficient_exp_a2 = 1/(10*Interfase_Val_a2)
    R_coefficient_exp_b2 = 1/(10*Interfase_Val_b2)
    R_coefficient_exp_c2 = 1/(10*Interfase_Val_c2)

    #surface strain thermal
#    if get(h.Compressed_strain,'value') == 1: #compressed
#        ISD_a = - 2 * Strain_Val_a * np.exp(-R_coefficient_exp_a * x_ISD)
#        ISD_b = - 2 * Strain_Val_b * np.exp(-R_coefficient_exp_b * x_ISD)
#        ISD_c = - 2 * Strain_Val_c * np.exp(-R_coefficient_exp_c * x_ISD)

#    else: #expansive strain
    ISD_a = @. 2 * Strain_Val_a * exp(-R_coefficient_exp_a * x_ISD)
    ISD_b = @. 2 * Strain_Val_b * exp(-R_coefficient_exp_b * x_ISD)
    ISD_c = @. 2 * Strain_Val_c * exp(-R_coefficient_exp_c * x_ISD)


#####For the future is the laser hits in the back of the crystal
    #surface strain thermal
#    if get(h.Compressed_strain2,'value') == 1: #compressed
#        ISD_a = ISD_a - 2*Strain_Val_a2 * np.exp(-R_coefficient_exp_a2 * flip(x_ISD))
#        ISD_b = ISD_b - 2*Strain_Val_b2 * np.exp(-R_coefficient_exp_b2 * flip(x_ISD))
#        ISD_c = ISD_c - 2*Strain_Val_c2 * np.exp(-R_coefficient_exp_c2 * flip(x_ISD))
#    else: #expansive strain
#        ISD_a = ISD_a + 2*Strain_Val_a2 * np.exp(-R_coefficient_exp_a2 * flip(x_ISD))
#        ISD_b = ISD_b + 2*Strain_Val_b2 * np.exp(-R_coefficient_exp_b2 * flip(x_ISD))
#        ISD_c = ISD_c + 2*Strain_Val_c2 * np.exp(-R_coefficient_exp_c2 * flip(x_ISD))


    #Bipolar waves
    #First wave
    R_coefficient_a = Interfase_Val_a/2
    R_coefficient_b = Interfase_Val_b/2
    R_coefficient_c = Interfase_Val_c/2

    #second wave
    R_coefficient_a2 = Interfase_Val_a2/2
    R_coefficient_b2 = Interfase_Val_b2/2
    R_coefficient_c2 = Interfase_Val_c2/2

    # Bipolar wave:
    if Compressed_strain1 #compressed
        ISD_a = @. ISD_a - Strain_Val_a * exp(- (x_ISD - cdepth_a + Interfase_Val_a/2)^2 / (2 * R_coefficient_a^2))
        ISD_a = @. ISD_a + Strain_Val_a * exp(- (x_ISD - cdepth_a - Interfase_Val_a/2)^2 / (2 * R_coefficient_a^2))
        ISD_b = @. ISD_b - Strain_Val_b * exp(- (x_ISD - cdepth_b + Interfase_Val_b/2)^2 / (2 * R_coefficient_b^2))
        ISD_b = @. ISD_b + Strain_Val_b * exp(- (x_ISD - cdepth_b - Interfase_Val_b/2)^2 / (2 * R_coefficient_b^2))
        ISD_c = @. ISD_c - Strain_Val_c * exp(- (x_ISD - cdepth_c + Interfase_Val_c/2)^2 / (2 * R_coefficient_c^2))
        ISD_c = @. ISD_c + Strain_Val_c * exp(- (x_ISD - cdepth_c - Interfase_Val_c/2)^2 / (2 * R_coefficient_c^2))
    else #expansive strain
        ISD_a = @. ISD_a + Strain_Val_a * exp(- (x_ISD - cdepth_a + Interfase_Val_a/2)^2 / (2 * R_coefficient_a^2))
        ISD_a = @. ISD_a - Strain_Val_a * exp(- (x_ISD - cdepth_a - Interfase_Val_a/2)^2 / (2 * R_coefficient_a^2))
        ISD_b = @. ISD_b + Strain_Val_b * exp(- (x_ISD - cdepth_b + Interfase_Val_b/2)^2 / (2 * R_coefficient_b^2))
        ISD_b = @. ISD_b - Strain_Val_b * exp(- (x_ISD - cdepth_b - Interfase_Val_b/2)^2 / (2 * R_coefficient_b^2))
        ISD_c = @. ISD_c + Strain_Val_c * exp(- (x_ISD - cdepth_c + Interfase_Val_c/2)^2 / (2 * R_coefficient_c^2))
        ISD_c = @. ISD_c - Strain_Val_c * exp(- (x_ISD - cdepth_c - Interfase_Val_c/2)^2 / (2 * R_coefficient_c^2))
    end

    #second wave
    if Compressed_strain2 #compressed
        ISD_a = @. ISD_a - Strain_Val_a2 * exp(- (x_ISD - cdepth_a2 + Interfase_Val_a2/2)^2 / (2 * R_coefficient_a2^2))
        ISD_b = @. ISD_b - Strain_Val_b2 * exp(- (x_ISD - cdepth_b2 + Interfase_Val_b2/2)^2 / (2 * R_coefficient_b2^2))
        ISD_c = @. ISD_c - Strain_Val_c2 * exp(- (x_ISD - cdepth_c2 + Interfase_Val_c2/2)^2 / (2 * R_coefficient_c2^2))
        ISD_a = @. ISD_a + Strain_Val_a2 * exp(- (x_ISD - cdepth_a2 - Interfase_Val_a2/2)^2 / (2 * R_coefficient_a2^2))
        ISD_b = @. ISD_b + Strain_Val_b2 * exp(- (x_ISD - cdepth_b2 - Interfase_Val_b2/2)^2 / (2 * R_coefficient_b2^2))
        ISD_c = @. ISD_c + Strain_Val_c2 * exp(- (x_ISD - cdepth_c2 - Interfase_Val_c2/2)^2 / (2 * R_coefficient_c2^2))
    else #expansive strain
        ISD_a = @. ISD_a + Strain_Val_a2 * exp(- (x_ISD - cdepth_a2 + Interfase_Val_a2/2)^2 / (2 * R_coefficient_a2^2))
        ISD_b = @. ISD_b + Strain_Val_b2 * exp(- (x_ISD - cdepth_b2 + Interfase_Val_b2/2)^2 / (2 * R_coefficient_b2^2))
        ISD_c = @. ISD_c + Strain_Val_c2 * exp(- (x_ISD - cdepth_c2 + Interfase_Val_c2/2)^2 / (2 * R_coefficient_c2^2))
        ISD_a = @. ISD_a - Strain_Val_a2 * exp(- (x_ISD - cdepth_a2 - Interfase_Val_a2/2)^2 / (2 * R_coefficient_a2^2))
        ISD_b = @. ISD_b - Strain_Val_b2 * exp(- (x_ISD - cdepth_b2 - Interfase_Val_b2/2)^2 / (2 * R_coefficient_b2^2))
        ISD_c = @. ISD_c - Strain_Val_c2 * exp(- (x_ISD - cdepth_c2 - Interfase_Val_c2/2)^2 / (2 * R_coefficient_c2^2))
    end

    ISD_steps = step_layer * ones(n_steps_layers)
    ISD_steps *= 1e-4

    return (; x_ISD, ISD_a, ISD_b, ISD_c, thickness_strain=ISD_steps)
end

function compute_beam(fwhm_x=0.5e-6, fwhm_y=0.5e-6; steps_x=40, steps_y=1000, I_range_x = 1, I_range_y = 500, yshift=0)
    # Definition of the two Gaussians for x and y of the incomming beam
    

    x0_array = 0
    y0_array = 0

    um_per_px_y = I_range_y / steps_y
    um_per_px_x = I_range_x / steps_x

    x_array = range(-I_range_x/2, I_range_x/2, steps_x) .* 10^-6
    y_array = range(-I_range_y/2, I_range_y/2, steps_y) .* 10^-6
    y_array = circshift(y_array, yshift)
    sigma_x = fwhm_x / 2.355
    sigma_y = fwhm_y / 2.355

    Gaussian_x = @. exp(-((x_array - x0_array)/sigma_x)^2 / 2)
    Gaussian_y = @. exp(-((y_array - y0_array)/sigma_y)^2 / 2)

    # Calculation of the ky_array and the kx_array vectors
    dx = x_array[2] - x_array[1]
    dy = y_array[2] - y_array[1]

    dkx = 2*pi/(steps_x * dx)
    dky = 2*pi/(steps_y * dy)

    kx_array = dkx * range(-steps_x/2, steps_x/2, steps_x)
    ky_array = dky * range(-steps_y/2, steps_y/2, steps_y)

    Gaussian_kx = FFTW.fftshift(FFTW.fft(Gaussian_x))
    Gaussian_ky = FFTW.fftshift(FFTW.fft(Gaussian_y))

    return (; x_array, y_array,
            Gaussian_x, Gaussian_y,
            kx_array, ky_array,
            Gaussian_kx, Gaussian_ky,
            um_per_px_x, um_per_px_y)
end

"""
Function to calculate the strain profile following the Thomsen paper: PRB 34 6 1986
Following Lings JPCM 18 2006 9231

Variables:
element = 'Ge';
Thickness_Crystal_um = 300; %um 300um
Thickness_Crystal_nostrain_um = 250; %um 200um of no strain
step_in_depth_um = 0.1;%um 100 nm
Q_l = 4e-3; % J/cm2 4 mJ/cm2 Energy laser
abs_depth_l_nm = 200;% 200nm for Ge with 800nm
laser_wl = 800; %Laser wavelenght in nm
Factor_tanh =1e9, %It is a factor that it is use for the tanh approximation
R_l = 33 %reflectivity for Si

Time variables:
time_step_ps = 8000;  %time is in ps after laser hit
"""
function Thomsen_model(; thickness=300, Thickness_Crystal_strain_um=50,
                       step_in_depth_um=0.1, laser_wl=800, Q_l=4e-3, R_l=33,
                       abs_depth_l_nm=200, Factor_tanh=1e9, time_step_ps=800,
                       v_sl=8433, v_st=5800, factor_lt=1,Anisotropic = false,
                       orthogonal_strain_zero = false, bipolar_trans = false, compressive = true)
    # From um to m
    Thickness_Crystal_um = thickness * 1e-6
    Thickness_Crystal_strain = Thickness_Crystal_strain_um * 1e-6
    Thickness_Crystal_nostrain = Thickness_Crystal_um - Thickness_Crystal_strain
    step_in_depth = step_in_depth_um * 1e-6

    # Depth crystal
    steps_depth = Thickness_Crystal_strain / step_in_depth
    n_steps = round(Int, steps_depth) + 1
    z = collect(range(step_in_depth, Thickness_Crystal_strain, n_steps))

    # Thickness_Crystal_strain_um
    thickness_strain = step_in_depth * ones(n_steps)

    # Laser parameters conversion
    hc = 1239.8
    E_p = hc / laser_wl     #E_p = 1.5497;%for 800nm
    ADL = abs_depth_l_nm * 1e-9 #Absoption depth to cm from nm

    # Time to s
    t_step = time_step_ps * 1e-12

    # Si
    C₁ = 1.66 #J/K/cm3 Si
    # vₛ = 8433 # m/s Speed sound Si m/s longitudinal
    #v_s = 5800 # m/s Speed sound Si m/s transversal

    #Anisotrpy 
    C11 = 16.6e11 #dm/cm2
    C12 = 6.4e11 #dm/cm2
    # Lings
    ϕβ = 4.08e-5 #K-1 Si %relates to the  linear expansion coeficient
    E_g  = 1.12 #eV Si

    # Thomsen
    Cₘ = 700 #J/kg/K Si
    ρ =  2.33 #g/cm3 Si
    Cᵥ = Cₘ * 1e-3 * ρ #J/K/cm3 Specific heat per unit volume
    β = 7.5e-6 #K-1  the  linear expansion coeficient Si
    η =  0.27 #Poisson ratio Si International System

    strain_f = @. Q_l * ϕβ * (E_p - E_g) / (100 * ADL * C₁ * E_p) #strain pre factor
    # Perpendicular
    strainTH_per = @. strain_f * (exp(-z / ADL) - 0.5 * (exp(-(z + v_sl * t_step)/ADL) + exp(-abs(z - v_sl * t_step) / ADL) * tanh((z - v_sl * t_step) * Factor_tanh)))
    
    # Parallel
    if !Anisotropic
        #Expansive 
        if !bipolar_trans
            if compressive
                strainTH_par = @. factor_lt * strain_f * (exp(-z / ADL) - 0.5 * (exp(-(z + v_st * t_step)/ADL) + exp(-abs(z - v_st * t_step) / ADL)))
            else #expansive
                strainTH_par = @. factor_lt * strain_f * (exp(-z / ADL) - 0.5 * (exp(-(z + v_st * t_step)/ADL) - exp(-abs(z - v_st * t_step) / ADL)))
            end
        else
            if compressive
                strainTH_par = @. factor_lt * strain_f * (exp(-z / ADL) - 0.5 * (exp(-(z + v_st * t_step)/ADL) + exp(-abs(z - v_st * t_step) / ADL) * tanh((z - v_st * t_step) * Factor_tanh)))
            else
                strainTH_par = @. factor_lt * strain_f * (exp(-z / ADL) - 0.5 * (exp(-(z + v_st * t_step)/ADL) - exp(-abs(z - v_st * t_step) / ADL) * tanh((z - v_st * t_step) * Factor_tanh)))
            end
        end
    else
        strainTH_par = @. - 2 * C12 / C11 * strainTH_per
    end

    if !orthogonal_strain_zero
        strainTH_par = @. 1 * strainTH_par
    else
        strainTH_par = @. 0 * strainTH_par
    end
    

    if t_step <= 0
        strainTH_per = strainTH_per.*0
        strainTH_par = strainTH_par .*0
    end


    
    # Add the layer non strain
    strainTH_per = push!(strainTH_per, 0)
    strainTH_par = push!(strainTH_par, 0)
    push!(z, z[end] + Thickness_Crystal_nostrain)

    thickness_strain[end] = Thickness_Crystal_nostrain
    thickness_strain .*= 1e6

    ISD_a = strainTH_per
    ISD_b = strainTH_par
    ISD_c = strainTH_par
    x_ISD = z

    return (; x_ISD, ISD_a, ISD_b, ISD_c, thickness_strain)
end

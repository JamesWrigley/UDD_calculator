# Contains re-useable buffers for the simulation
struct SimBuffers{ComplexVec, ComplexTensor, FloatVec, FloatTensor, Plan}
    R_00_S0::ComplexTensor
    R_0H_S0::ComplexTensor
    r_s_g::ComplexTensor
    scratch::ComplexTensor
    alfa::FloatTensor
    ifftplan::Plan

    Gaussian_kx::ComplexVec
    Gaussian_ky::ComplexVec
    k0_Theta::FloatVec
end

function SimBuffers(shape; ArrayT=CuArray, ComplexT=ComplexF64, FloatT=Float64)
    R_00_S0 = ArrayT{ComplexT, 3}(undef, shape)
    R_0H_S0 = similar(R_00_S0)
    r_s_g = similar(R_00_S0)
    scratch = similar(R_00_S0)
    alfa = ArrayT{FloatT, 3}(undef, shape)
    ifftplan = CUFFT.plan_ifft(r_s_g)

    Gaussian_kx = ArrayT{ComplexT, 1}(undef, shape[1])
    Gaussian_ky = ArrayT{ComplexT, 1}(undef, shape[2])
    k0_Theta = ArrayT{FloatT, 1}(undef, shape[3])

    buffers = SimBuffers(R_00_S0, R_0H_S0, r_s_g, scratch, alfa, ifftplan,
                         Gaussian_kx, Gaussian_ky, k0_Theta)
end

const default_buffers = Ref{SimBuffers}()

@kernel function compute_r!(i_layers, b, @Const(alfa), Chi_0_Cx, q, Thickness,
                            Chi_0_factor, Chi_h_n_factor, k0_Bragg_gam_factor,
                            # These arguments will be modified
                            R_00_S0, R_0H_S0)
    i = @index(Global, Linear)

    y_i = @inbounds 0.5 * (b * alfa[i] + Chi_0_factor)
    q_y_sqrt_i = sqrt(q + y_i^2)

    R2_i = (-y_i + q_y_sqrt_i) * Chi_h_n_factor
    R1_i = (-y_i - q_y_sqrt_i) * Chi_h_n_factor

    x_1 = 0.5 * (Chi_0_Cx + (-y_i + q_y_sqrt_i))
    x_2 = 0.5 * (Chi_0_Cx + (-y_i - q_y_sqrt_i))
    expX1_i = exp(1im * k0_Bragg_gam_factor * x_1 * Thickness)
    expX2_i = exp(1im * k0_Bragg_gam_factor * x_2 * Thickness)

    # Multiplication is faster than division, so we take the reciprocal of
    # the difference here and multiply with that instead of doing divisions
    # by the difference.
    R_diff_recip = 1 / (R2_i - R1_i)
    if i_layers == 1
        @inbounds R_00_S0[i] = (R2_i * expX1_i - R1_i * expX2_i) * R_diff_recip
        @inbounds R_0H_S0[i] = R2_i * R1_i * (expX1_i - expX2_i) * R_diff_recip
    else
        @inbounds old_00 = R_00_S0[i]
        @inbounds old_0H = R_0H_S0[i]

        @inbounds R_00_S0[i] = (((R2_i * expX1_i - R1_i * expX2_i) * R_diff_recip) * old_00 +
                                ((expX2_i - expX1_i)               * R_diff_recip) * old_0H)
        @inbounds R_0H_S0[i] = ((R2_i * R1_i * (expX1_i - expX2_i) * R_diff_recip) * old_00 +
                                ((R2_i * expX2_i - R1_i * expX1_i) * R_diff_recip) * old_0H)
    end
end

@kernel function compute_alfa!(@Const(Matrix_Bragg),
                               @Const(kx_array), @Const(ky_array), @Const(k0_Theta),
                               m_d, alfa_strain_constant,
                               d_hkl_factor, k0_Bragg_factor,
                               alfa)
    # The shape of alfa is (x, y, t)
    i = @index(Global, Cartesian)
    x = @inbounds kx_array[i[1]]
    y = @inbounds ky_array[i[2]]
    t = @inbounds k0_Theta[i[3]]

    kz0_i = sqrt(t^2 - y^2 - x^2)
    k0_beam_j1 = @inbounds Matrix_Bragg[1, 1] * x + Matrix_Bragg[1, 2] * y + Matrix_Bragg[1, 3] * kz0_i
    k0_beam_j2 = @inbounds Matrix_Bragg[2, 1] * x + Matrix_Bragg[2, 2] * y + Matrix_Bragg[2, 3] * kz0_i
    k0_beam_j3 = @inbounds Matrix_Bragg[3, 1] * x + Matrix_Bragg[3, 2] * y + Matrix_Bragg[3, 3] * kz0_i

    alfa_beam_i = (@inbounds m_d[1] * k0_beam_j1 +
        @inbounds m_d[2] * k0_beam_j2 +
        @inbounds m_d[3] * k0_beam_j3) / 2 / pi

    @inbounds alfa[i] = (2alfa_beam_i + d_hkl_factor) * k0_Bragg_factor + alfa_strain_constant
end

@kernel function precompute_r_s_g!(r_s_g,
                                   @Const(R_0H_S0), @Const(Gaussian_kx), @Const(Gaussian_ky),
                                   @Const(k0_Theta), k0, sigk)
    i = @index(Global, Cartesian)
    x = @inbounds Gaussian_kx[i[1]]
    y = @inbounds Gaussian_ky[i[2]]
    t = @inbounds k0_Theta[i[3]]

    Gaussian_k = exp(-0.5 * (t - k0)^2 / sigk^2)
    Gaussian_K_3D = x * y * Gaussian_k

    @inbounds r_s_g[i] = R_0H_S0[i] * Gaussian_K_3D
end

@kernel function compute_k0_theta!(k0_Theta,
                                   N_Step, Theta_Initial, Steps_De_Theta,
                                   Ang_asy, d_hkl)
    i = @index(Global, Linear)

    Theta_De = Theta_Initial + (Steps_De_Theta * i)
    Theta = Theta_De * pi / 180 - Ang_asy #Back to rad

    # Calculation of the Energy for each angle
    WaveL = 2 * d_hkl * sin(Theta)

    @inbounds k0_Theta[i] = 2 * pi / WaveL
end

function greens_postprocess(buffers)

end

function laue_strain(
    Energy_Bragg,
    hkl,
    DWF, sf,
    absor,
    Range_E_neg, Range_E_pos,
    Polarization,
    Ang_asy_Deg_strain, pulse,
    N_Step,
    beam,
    crystal_orientation;
    # This value should be in radians
    delta_theta_manual=0,
    T=CuArray,
    max_layers=Inf,
    plane::Union{Vector{Symbol}, Symbol}=[:forward, :diffracted],
    buffers_ref=default_buffers, progressbar=true)
    if plane isa Symbol
        plane = [plane]
    end

    # Perpendicular to the surface (100)
    strain_per = pulse.ISD_a

    # Parallel to surface (100)
    strain_par = pulse.ISD_b + pulse.ISD_c

    a_par_strain = sf.a_Par .+ pulse.ISD_a
    b_par_strain = sf.b_Par .+ pulse.ISD_b
    c_par_strain = sf.c_Par .+ pulse.ISD_c

    # Wave Length Bragg
    WaveL_Bragg = h_planck * c_light / Energy_Bragg #m
    k0 = 2 * pi / WaveL_Bragg
    sigk = rho * k0

    layers = length(pulse.thickness_strain) # number of layers

    # Structure factors
    i_F0 = sf.F0
    i_FH = sf.FH
    i_F_H = sf.F_H

    #Asymmetry
    i_Ang_asy_Deg = Ang_asy_Deg_strain

    output_shape = (length(beam.kx_array), length(beam.ky_array), N_Step)

    if !isassigned(buffers_ref)
        buffers_ref[] = SimBuffers(output_shape; ArrayT=T)
    elseif size(buffers_ref[].R_00_S0) != output_shape
        buffers_ref[] = SimBuffers(output_shape; ArrayT=T)
    end

    buffers = buffers_ref[]

    copy!(buffers.Gaussian_kx, beam.Gaussian_kx)
    copy!(buffers.Gaussian_ky, beam.Gaussian_ky)

    backend = KA.get_backend(buffers.R_00_S0)
    r_kernel! = compute_r!(backend)
    alfa_kernel! = compute_alfa!(backend)
    r_s_g_kernel! = precompute_r_s_g!(backend)
    k0_theta_kernel! = compute_k0_theta!(backend)

    end_layer = Int(min(layers, max_layers))
    p = Progress(end_layer; enabled=progressbar == true, showspeed=true)
    if progressbar isa Base.RefValue
        progressbar[] = p
    end

    for i_layers in 1:end_layer
        #Crystal parameters
        Thickness = pulse.thickness_strain[i_layers] * 1e-6

        #Expansion of the unit cell
        i_a_Par = a_par_strain[i_layers]
        i_b_Par = b_par_strain[i_layers]
        i_c_Par = c_par_strain[i_layers]

        d_hkl= 10^-10 / sqrt((hkl.h / i_a_Par)^2 +
            (hkl.k / i_b_Par)^2 +
            (hkl.l / i_c_Par)^2)

        #Trying minimize the effects of exp
        r_e_over_V = 2.8179403267e15 / (i_a_Par*i_b_Par*i_c_Par)

        #Calculation of the permeability
        absor_F0 = absor ? sf.F0 : abs(sf.F0)
        absor_FH = absor ? sf.FH : abs(sf.FH)
        absor_F_H = absor ? sf.F_H : abs(sf.F_H)
        Chi_0_Cx   = r_e_over_V * WaveL_Bragg^2 * absor_F0  / pi
        Chi_h_Cx   = r_e_over_V * WaveL_Bragg^2 * absor_FH  / pi * DWF
        Chi_h_n_Cx = r_e_over_V * WaveL_Bragg^2 * absor_F_H / pi * DWF

        #Beam properties
        k0_Bragg = 2 * pi / WaveL_Bragg

        Theta_Bragg = asin(WaveL_Bragg / (2 * d_hkl)) + delta_theta_manual

        #Definition polarization and Asymmetry
        P = Polarization === :p ? cos(2 * Theta_Bragg) : 1.0

        Ang_asy = Ang_asy_Deg_strain * pi / 180

        #Cosine directers
        gam_0 = sin(pi/2 - Theta_Bragg + Ang_asy)
        gam_H = sin(pi/2 + Theta_Bragg - Ang_asy)

        b = gam_0 / gam_H

        m_d = T([0, cos(Ang_asy)/d_hkl, sin(Ang_asy)/d_hkl])

        q = b * Chi_h_Cx * Chi_h_n_Cx * abs(P)^2

        #Range work
        #In case there is assymetry
        Theta_Bragg_Asy = Theta_Bragg + Ang_asy

        WaveL_Bragg_neg = h_planck*c_light / (Energy_Bragg - Range_E_neg) #m
        WaveL_Bragg_pos = h_planck*c_light / (Energy_Bragg + Range_E_pos) #

        Theta_Bragg_neg = asin(WaveL_Bragg_neg / (2*d_hkl)) + Ang_asy
        Theta_Bragg_pos = asin(WaveL_Bragg_pos / (2*d_hkl)) + Ang_asy

        Range_De_neg = Theta_Bragg_neg * 180 / pi #to Deg
        Range_De_pos = Theta_Bragg_pos * 180 / pi #to deg

        Theta_Initial = Range_De_neg
        Steps_De_Theta = (Range_De_pos - Range_De_neg) / N_Step

        if crystal_orientation ##+ diffraction
            Matrix_Bragg = [[1 0                    0]
                            [0 cos(Theta_Bragg_Asy) -sin(Theta_Bragg_Asy)]
                            [0 sin(Theta_Bragg_Asy) cos(Theta_Bragg_Asy)]]
        else  ##- diffraction
            Matrix_Bragg = [[-1 0                    0]
                            [0 -cos(Theta_Bragg_Asy) -sin(Theta_Bragg_Asy)]
                            [0 sin(Theta_Bragg_Asy)  -cos(Theta_Bragg_Asy)]]
        end
        Matrix_Bragg = T(Matrix_Bragg)

        # Vectorized version of the code
        i_Theta = 1:N_Step

        Theta_De = @. Theta_Initial + (Steps_De_Theta * i_Theta)
        Theta = @. Theta_De * pi / 180 - Ang_asy #Back to rad

        #Calculation of the Energy for each angle
        WaveL = T(@. 2 * d_hkl * sin(Theta))

        @. buffers.k0_Theta = 2 * pi / WaveL

        # Both perpendicular and parallel strain
        if all(Theta .> Theta_Bragg)
            alfa_strain_constant = (sin(Ang_asy)^2 * tan(Theta_Bragg) - sin(Ang_asy) * cos(Ang_asy)) * strain_per[i_layers] +
                                   (cos(Ang_asy)^2 * tan(Theta_Bragg) - sin(Ang_asy) * cos(Ang_asy)) * strain_par[i_layers]
        else
            alfa_strain_constant = (sin(Ang_asy)^2 * tan(Theta_Bragg) + sin(Ang_asy) * cos(Ang_asy)) * strain_per[i_layers] +
                                   (cos(Ang_asy)^2 * tan(Theta_Bragg) + sin(Ang_asy) * cos(Ang_asy)) * strain_par[i_layers]
        end

        d_hkl_factor = 1 / d_hkl^2
        k0_Bragg_factor = 4pi^2 / k0_Bragg^2
        alfa_kernel!(Matrix_Bragg, beam.kx_array, beam.ky_array, buffers.k0_Theta,
                     m_d, alfa_strain_constant,
                     d_hkl_factor, k0_Bragg_factor,
                     buffers.alfa, ndrange=size(buffers.alfa))

        # Precompute some factors
        Chi_0_factor = Chi_0_Cx * (1 - b)
        Chi_h_n_factor = 1 / (Chi_h_n_Cx * P)
        k0_Bragg_gam_factor = k0_Bragg / gam_0
        r_kernel!(i_layers, b, buffers.alfa, Chi_0_Cx, q, Thickness,
                  Chi_0_factor, Chi_h_n_factor, k0_Bragg_gam_factor,
                  buffers.R_00_S0, buffers.R_0H_S0,
                  ndrange=size(buffers.R_00_S0))
        KA.synchronize(backend)
        next!(p)
    end

    diffracted_x_profile = nothing
    diffracted_y_profile = nothing
    diffracted_mode_gauss = nothing
    diffracted_phase = nothing
    forward_mode_gauss = nothing
    forward_x_profile = nothing
    forward_y_profile = nothing
    forward_phase = nothing
    for p in plane
        greens_function = p == :diffracted ? buffers.R_0H_S0 : buffers.R_00_S0
        r_s_g_kernel!(buffers.r_s_g,
              greens_function, buffers.Gaussian_kx, buffers.Gaussian_ky,
              buffers.k0_Theta, k0, sigk,
              ndrange=size(buffers.r_s_g))
        
        CUFFT.fftshift!(buffers.scratch, buffers.r_s_g)
        mul!(buffers.r_s_g, buffers.ifftplan, buffers.scratch)
        gaussian_r = CUFFT.fftshift!(buffers.scratch, buffers.r_s_g, (2, 3))
     
        gaussian_r_2d = squeeze(sum(gaussian_r, dims=3))
        # circshift() to fully shift the signal. The plain fftshift()
        # still leaves a few elements wrapped around the end of the array.
        gaussian_r_2d = circshift(gaussian_r_2d, (0, 20))
        mode_gauss_gpu = abs.(gaussian_r_2d)
        phase_gauss_gpu = angle.(gaussian_r_2d)
        y_profile = squeeze(Array(sum(mode_gauss_gpu, dims=1)))
        x_profile = squeeze(Array(sum(mode_gauss_gpu, dims=2)))

        mode_gauss = permutedims(Array(mode_gauss_gpu), (2, 1))
        phase_gauss = permutedims(Array(phase_gauss_gpu), (2, 1))
        
        if p == :diffracted
            diffracted_mode_gauss = mode_gauss
            diffracted_x_profile = x_profile
            diffracted_y_profile = y_profile
            diffracted_phase = phase_gauss
        else
            forward_mode_gauss = mode_gauss
            forward_x_profile = x_profile
            forward_y_profile = y_profile
            forward_phase = phase_gauss
        end
    end

    # fftshift all dimensions, inverse FFT, and then fftshift again over the y
    # and t dimensions.



    # Swap the dimensions of mode_gaus for simpler plotting
    results = (; # R_0H_S0=Array(buffers.R_0H_S0),
               # R_00_S0=Array(buffers.R_00_S0),
               # k0_Theta=Array(buffers.k0_Theta),
               mean_reflectance=mean(buffers.R_0H_S0),
               diffracted_mode_gauss,
               diffracted_y_profile,
               diffracted_x_profile,
               diffracted_phase,
               forward_mode_gauss,
               forward_y_profile,
               forward_x_profile,
               forward_phase)

    return results
end

"""
Slow but simple reference implementation of the simulation.
"""
function laue_strain_reference(
    a_Par_strain, b_Par_strain, c_Par_strain,
    Energy_Bragg,
    Energy_center_strain, hkl,
    DWF, sf,
    absor,
    Range_E_neg, Range_E_pos,
    Polarization,
    Ang_asy_Deg_strain, pulse,
    N_Step,
    ky_array, kx_array,
    crystal_orientation,
    strain_per,strain_par)
    # Constants
    c_light= 299792458                   #Light Speed m/s
    h_planck = 4.13566733*10^-15         #eV

    #h_plank and c_light
    hc = 1.2398e-06;

    r_e = 2.8179403267*10^-15

    # Calculations
    #Wave Length Bragg
    WaveL_Bragg = h_planck * c_light / Energy_Bragg #m

    layers = length(pulse.thickness_strain) # number of layers

    #print(layers)
    #time_past = 0;

    #Miller index of the reflection
    i_h_Miller = hkl[1]
    i_k_Miller = hkl[2]
    i_l_Miller = hkl[3]

    #Structure factors
    i_F0 = sf.F0
    i_FH = sf.FH
    i_F_H = sf.F_H

    #Central energy of the beam
    i_Energy_center = Energy_center_strain

    #Asymmetry
    i_Ang_asy_Deg = Ang_asy_Deg_strain

    #Loop all layers
    R_00_S0 = nothing
    R_0H_S0 = nothing
    local k0_Theta
    for i_layers in 1:layers
        tic = time()
        @info "Computing layer $(i_layers)..."
        #Crystal parameters
        Thickness = pulse.thickness_strain[i_layers] * 1e-6

        #Expansion of the unit cell
        i_a_Par = a_Par_strain[i_layers]
        i_b_Par = b_Par_strain[i_layers]
        i_c_Par = c_Par_strain[i_layers]

        #Calculation of the matrix
        V = i_a_Par*10^-10 * i_b_Par*10^-10 * i_c_Par*10^-10;

        d_hkl= 10^-10 / sqrt((i_h_Miller/(i_a_Par))^2 + (i_k_Miller/(i_b_Par))^2 + (i_l_Miller/(i_c_Par))^2)

        #Trying minimize the effects of exp
        r_e_over_V = 2.8179403267*10^15 / (i_a_Par*i_b_Par*i_c_Par)

        #Calculation of the permeability
        if !absor
            Chi_0_Cx   = r_e_over_V * WaveL_Bragg^2 * abs(sf.F0) /  pi
            Chi_h_Cx   = r_e_over_V * WaveL_Bragg^2 * abs(sf.FH) /  pi * DWF
            Chi_h_n_Cx = r_e_over_V * WaveL_Bragg^2 * abs(sf.F_H) / pi * DWF
        else
            Chi_0_Cx   = r_e_over_V * WaveL_Bragg^2 * sf.F0 /  pi
            Chi_h_Cx   = r_e_over_V * WaveL_Bragg^2 * sf.FH /  pi * DWF
            Chi_h_n_Cx = r_e_over_V * WaveL_Bragg^2 * sf.F_H / pi * DWF
        end

        #Beam properties
        k0_Bragg = 2 * pi / WaveL_Bragg

        Theta_Bragg = asin(WaveL_Bragg / (2 * d_hkl))

        #Definition polarization and Asymmetry
        if Polarization === :p
            P = cos(2 * Theta_Bragg)
        else
            P = 1
        end

        Ang_asy = Ang_asy_Deg_strain * pi / 180

        #Cosine directers
        gam_0 = sin(pi/2 - Theta_Bragg + Ang_asy)
        gam_H = sin(pi/2 + Theta_Bragg - Ang_asy)

        b = gam_0 / gam_H

        m_d = [0, cos(Ang_asy)/d_hkl, sin(Ang_asy)/d_hkl]

        q = b * Chi_h_Cx * Chi_h_n_Cx * abs(P)^2

        #Range work
        #In case there is assymetry
        Theta_Bragg_Asy = Theta_Bragg + Ang_asy

        WaveL_Bragg_neg = h_planck*c_light / (Energy_Bragg - Range_E_neg) #m
        WaveL_Bragg_pos = h_planck*c_light / (Energy_Bragg + Range_E_pos) #

        Theta_Bragg_neg = asin(WaveL_Bragg_neg / (2*d_hkl)) + Ang_asy
        Theta_Bragg_pos = asin(WaveL_Bragg_pos / (2*d_hkl)) + Ang_asy

        Range_De_neg = Theta_Bragg_neg * 180 / pi #to Deg
        Range_De_pos = Theta_Bragg_pos * 180 / pi #to deg

        Theta_Initial = Range_De_neg
        Steps_De_Theta = (Range_De_pos - Range_De_neg) / N_Step

        if crystal_orientation ##+ diffraction
            Matrix_Bragg = [[1 0                    0]
                            [0 cos(Theta_Bragg_Asy) -sin(Theta_Bragg_Asy)]
                            [0 sin(Theta_Bragg_Asy) cos(Theta_Bragg_Asy)]]
        else  ##- diffraction
            Matrix_Bragg = [[-1 0                    0]
                            [0 -cos(Theta_Bragg_Asy) -sin(Theta_Bragg_Asy)]
                            [0 sin(Theta_Bragg_Asy)  -cos(Theta_Bragg_Asy)]]
        end

        # Vectorized version of the code
        i_Theta = 1:N_Step

        Theta_De = @. Theta_Initial + (Steps_De_Theta * i_Theta) #i think i do a vector
        Theta = @. Theta_De * pi / 180 - Ang_asy #Back to rad

        #Calculation of the Energy for each angle
        WaveL = @. 2 * d_hkl * sin(Theta)

        Energy = @. hc / WaveL #hc is a constatant to transfor energy en wavelength

        #We save the value of the diference of the energy in an array
        E_Scan =  @. Energy - Energy_Bragg

        k0_Theta = @. 2 * pi / WaveL

        #########################
        ##CODE###
        #################################################################################################
        kx_arrayr = reshape(kx_array, (length(kx_array), 1, 1))
        ky_arrayr = reshape(ky_array, (1, length(ky_array), 1))
        k0_Thetar = reshape(k0_Theta, (1, 1, length(k0_Theta)))

        kz0 = @. sqrt(k0_Thetar^2 - ky_arrayr^2 - kx_arrayr^2)

        k0_x = @views Matrix_Bragg[:, 1][:, newaxis, newaxis, newaxis] .* kx_arrayr[newaxis, :, :, :]
        k0_y = @views Matrix_Bragg[:, 2][:, newaxis, newaxis, newaxis] .* ky_arrayr[newaxis, :, :, :]
        k0_z = @views Matrix_Bragg[:, 3][:, newaxis, newaxis, newaxis] .* kz0[newaxis, :, :, :]
        k0_beam = @. k0_x + k0_y + k0_z

        alfa_beam = (m_d[1] * k0_beam[1, :, :, :] +
            m_d[2] * k0_beam[2, :, :, :] +
            m_d[3] * k0_beam[3, :, :, :]) / 2 / pi

        if all(Theta .> Theta_Bragg)
            # Both perpendicular and parallel strain
            alfa = @. ((2alfa_beam + 1 / d_hkl^2) * 4pi^2 / k0_Bragg^2 +
                (sin(Ang_asy)^2 * tan(Theta_Bragg) - sin(Ang_asy) * cos(Ang_asy)) * strain_per[i_layers] +
                (cos(Ang_asy)^2 * tan(Theta_Bragg) - sin(Ang_asy) * cos(Ang_asy)) * strain_par[i_layers])
        else
            # Both perpendicular and parallel strain
            alfa = @. ((2alfa_beam + 1 / d_hkl^2) * 4pi^2 / k0_Bragg^2 +
                (sin(Ang_asy)^2 * tan(Theta_Bragg) + sin(Ang_asy) * cos(Ang_asy)) * strain_per[i_layers] +
                (cos(Ang_asy)^2 * tan(Theta_Bragg) + sin(Ang_asy) * cos(Ang_asy)) * strain_par[i_layers])
        end

        # Definition of y
        y = @. 0.5 * (b * alfa + Chi_0_Cx * (1 - b))
        q_y_sqrt = @. sqrt(q + y^2)

        R2 = @. (-y + q_y_sqrt) / (Chi_h_n_Cx * P)
        R1 = @. (-y - q_y_sqrt) / (Chi_h_n_Cx * P)
        R_diff = R2 - R1

        X_1 = @. 0.5 * (Chi_0_Cx + (-y + q_y_sqrt))
        X_2 = @. 0.5 * (Chi_0_Cx + (-y - q_y_sqrt))

        expX1 = @. exp(1im * k0_Bragg / gam_0 * X_1 * Thickness)
        expX2 = @. exp(1im * k0_Bragg / gam_0 * X_2 * Thickness)

        if i_layers == 1
            # First layer
            R_00_S0 = @. (R2 * expX1 - R1 * expX2) / R_diff
            R_0H_S0 = @. R2 * R1 * (expX1 - expX2) / R_diff
        else
            # Next layers
            R_00_S = @. (((R2 * expX1 - R1 * expX2)  / R_diff) * R_00_S0 +
                ((expX2 - expX1)            / R_diff) * R_0H_S0)
            R_0H_S = @. ((R2 * R1 * (expX1 - expX2)  / R_diff) * R_00_S0 +
                ((R2 * expX2 - R1 * expX1)  / R_diff) * R_0H_S0)
            R_00_S0 = R_00_S
            R_0H_S0 = R_0H_S
        end
    end

    return R_0H_S0, R_00_S0, k0_Theta
end

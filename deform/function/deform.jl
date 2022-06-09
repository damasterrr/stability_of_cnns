module deform

function deform(original, m) #m recommended from 0.1 to 1
    
    nfreq = 4
    as  = randn(nfreq, nfreq)
    aas = randn(nfreq, nfreq)
    bs  = randn(nfreq, nfreq)
    bbs = randn(nfreq, nfreq)
    
    f1(x,y) = sum( as[i,j]*sinpi(x*i+ aas[i,j])*sinpi(y*j )/(i^2+j^2)^(1.5) for i=1:nfreq, j=1:nfreq)
    f2(x,y) = sum( bs[i,j]*sinpi(x*i)*sinpi(y*j + bbs[i,j])/(i^2+j^2)^(1.5) for i=1:nfreq, j=1:nfreq)

    
    s = size(original)[1]
    
    original2 = copy(original)
    
    function move_24(x_0, y_0, m, cycle_len)
        for _ in 1:cycle_len
            x_0 = clamp(f1(x_0,y_0)*m + x_0, 0, 1)
            y_0 = clamp(f2(x_0,y_0)*m + y_0, 0, 1) 
        end
        return x_0,y_0
    end
    
    for i in range(0.01,0.99,step=0.5/s)
        for j in range(0.01,0.99,step=0.5/s)
            a = floor.(Int, move_24(i,j,m,15) .* s) # <----------------- number of integrations
            #println(a)
            for k in range(1,3,step=1)

                original2[clamp((a[2]), 1, s),clamp((a[1]), 1, s),k] = original[clamp(floor(Int, j*s),1,s),clamp(floor(Int, i*s),1,s),k]
                #deformed[clamp((a[2]), 1, s),clamp((a[1]), 1, s),None] = original[clamp(floor(Int, j*s),1,s),clamp(floor(Int, i*s),1,s),None]
            end
        end   
    end
    
    return original2
end

end # module

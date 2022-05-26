import Pkg; Pkg.add("PyPlot"); Pkg.add("OrdinaryDiffEq"); Pkg.add("FileIO"); Pkg.add("Images"); Pkg.add("ImageAxes"); Pkg.add("Colors");  Pkg.add("Plots"); Pkg.add("TestImages"); Pkg.add("MLDatasets"); Pkg.add("PyCall"); Pkg.add("StatsBase")

nfreq = 4
as  = randn(nfreq, nfreq)
aas = randn(nfreq, nfreq)
bs  = randn(nfreq, nfreq)
bbs = randn(nfreq, nfreq)

as = repeat(as,inner = (3,3))
aas = repeat(aas,inner = (3,3))
bs = repeat(bs,inner = (3,3))
bbs = repeat(bbs,inner = (3,3))
        
using Images
using TestImages
img_lab = rand(Lab, 100, 100) #random pixel image

img_path1 = "Mandrill.png" #512x512 array
img_mandrill = load(img_path1)

img_path2 = "girl.png" #100x100 array
img_girl = load(img_path2)

img_path3 = "girl_256.png" #256x256
img_girl_256 = load(img_path3)

img_path4 = "girl_high_res.png" #512x512
img_girl_high = load(img_path4)

img_path5 = "grid_100.png" #512x512
img_grid = load(img_path5)

img_path5 = "square.png" #100x100
img_sq = load(img_path5)

img_path6 = "girl_framed.png" #100x100
img_girl_framed = load(img_path6)

img = imresize(parent(testimage("mandrill")),32,32)

function move_io_back(x_0, y_0, m)
    #scatter(x_0, y_0)
    x_1 = clamp(-f1(x_0,y_0)*m + x_0, 0, 1)
    y_1 = clamp(-f2(x_0,y_0)*m + y_0, 0, 1) 

    #scatter(x_1, y_1)

    x_2 = clamp(-f1(x_1,y_1)*m + x_1, 0, 1)
    y_2 = clamp(-f2(x_1,y_1)*m + y_1, 0, 1)
    #scatter(x_2, y_2)

    x_3 = clamp(-f1(x_2,y_2)*m + x_2, 0, 1)
    y_3 = clamp(-f2(x_2,y_2)*m + y_2, 0, 1)
    #scatter(x_3, y_3)
    
    x_4 = clamp(-f1(x_3,y_3)*m + x_3, 0, 1)
    y_4 = clamp(-f2(x_3,y_3)*m + y_3, 0, 1)
    #scatter(x_4, y_4)
    
    #plot([x_0,x_1,x_2,x_3],[y_0,y_1,y_2,y_3])
    return x_2,y_2
end

function check_bijection(restored, original)
    s = size(original)[1]
    b = Vector{Bool}()
    for i in range(1,s,step=1)
        for j in range(1,s,step=1)
            if restored[i,j] == original[i,j]
               push!(b, true)
            end           
        end
    end
    a = 100*(length(b) /(s*s))
    return print("Accuracy of diffeomorphism: $a %")
end

function restore(deformed, original, m) #m recommended from 0.1 to 1
    s = size(original)[1]
    
    restored = RGB.(Array(reshape(range(0,stop=0,length=10^4), s, s)))
    
    for i in range(0.01,0.99,step=0.5/s)
        for j in range(0.01,0.99,step=0.5/s)
            a = ceil.(Int, move_io_back(i,j,m) .* s)
            #println(a)

            restored[clamp((a[2]), 1, s),clamp((a[1]), 1, s)] = deformed[clamp(floor(Int, j*s),1,s),clamp(floor(Int, i*s),1,s)]
            
            #img_empty_back[floor(Int, j*s),floor(Int, i*s)] = img_empty[clamp((a[2]), 1, s),clamp((a[1]), 1, s)]
        end
    end
    check_bijection(restored, original)
    return mosaicview(restored; nrow=1)
end

function move_new(x_0, y_0, m, cycle_len)
    for _ in 1:cycle_len
        x_0 = clamp(f1(x_0,y_0)*m + x_0, 0, 1)
        y_0 = clamp(f2(x_0,y_0)*m + y_0, 0, 1) 
    end
    return x_0,y_0
end

using PyPlot

f1(x,y) = sum( as[i,j]*sinpi(x*i+ aas[i,j])*sinpi(y*j )/(i^2+j^2)^(1.5) for i=1:nfreq, j=1:nfreq)
f2(x,y) = sum( bs[i,j]*sinpi(x*i)*sinpi(y*j + bbs[i,j])/(i^2+j^2)^(1.5) for i=1:nfreq, j=1:nfreq)

function move(x_0, y_0, m) #probably can be written more compactly
    scatter(x_0, y_0)
    x_1 = clamp(f1(x_0,y_0)*m + x_0, 0, 1)
    y_1 = clamp(f2(x_0,y_0)*m + y_0, 0, 1)

    #scatter(x_1, y_1)

    x_2 = clamp(f1(x_1,y_1)*m + x_1, 0, 1)
    y_2 = clamp(f2(x_1,y_1)*m + y_1, 0, 1)
    #scatter(x_2, y_2)

    x_3 = clamp(f1(x_2,y_2)*m + x_2, 0, 1)
    y_3 = clamp(f2(x_2,y_2)*m + y_2, 0, 1)
    #scatter(x_3, y_3)
    
    x_4 = clamp(f1(x_3,y_3)*m + x_3, 0, 1)
    y_4 = clamp(f2(x_3,y_3)*m + y_3, 0, 1)
    #scatter(x_4, y_4)
    
    
    plot([x_0,x_1,x_2,x_3,x_4],[y_0,y_1,y_2,y_3,y_4])
end

#does not plot, just gives values, gives values for the image indexes
function move_io(x_0, y_0, m)
    #scatter(x_0, y_0)
    x_1 = clamp(f1(x_0,y_0)*m + x_0, 0, 1)
    y_1 = clamp(f2(x_0,y_0)*m + y_0, 0, 1) 

    #scatter(x_1, y_1)

    x_2 = clamp(f1(x_1,y_1)*m + x_1, 0, 1)
    y_2 = clamp(f2(x_1,y_1)*m + y_1, 0, 1)
    #scatter(x_2, y_2)

    x_3 = clamp(f1(x_2,y_2)*m + x_2, 0, 1)
    y_3 = clamp(f2(x_2,y_2)*m + y_2, 0, 1)
    #scatter(x_3, y_3)
    
    x_4 = clamp(f1(x_3,y_3)*m + x_3, 0, 1)
    y_4 = clamp(f2(x_3,y_3)*m + y_3, 0, 1)
    #scatter(x_4, y_4)
    
    #plot([x_0,x_1,x_2,x_3],[y_0,y_1,y_2,y_3])
    return x_2,y_2
end

#poitns
u = [[i,j] for i in 0:0.1:1, j in 0:0.1:1]

for j in range(1,121, step=1)
    move(u[j][2], 1 - u[j][1], 0.2)
end

#field
xs = range(0,1,step=0.05)
ys = range(0,1,step=0.05)
quiver(xs,ys, f1.(xs',ys), f2.(xs',ys))

function deform(original, m) #m recommended from 0.1 to 1
    s = size(original)[1]
    deformed = RGB.(Array(reshape(range(0,stop=0,length=s*s), s, s)))
    #original2 = copy(original)
    
    for i in range(0.01,0.99,step=0.5/s)
        for j in range(0.01,0.99,step=0.5/s)
            a = floor.(Int, move_new(i,j,m,2) .* s)
            #println(a)

            #original2[clamp((a[2]), 1, s),clamp((a[1]), 1, s)] = original[clamp(floor(Int, j*s),1,s),clamp(floor(Int, i*s),1,s)]
            deformed[clamp((a[2]), 1, s),clamp((a[1]), 1, s)] = original[clamp(floor(Int, j*s),1,s),clamp(floor(Int, i*s),1,s)]
        end
    end
    #restored = restore(original2, original, m)
    #return mosaicview(restored, original2, original; nrow=1)
    return deformed
end

deform(img_girl,0.1)

using Images, TestImages, Colors



function frame(img,l)
    s = size(img)[1]
    
    img_framed = padarray(img, Fill(colorant"magenta", (l, l), (l, l)))

    img2 = imresize(parent(img_framed), (s, s))
    
    return img2
    
end

frame(img_girl,10)


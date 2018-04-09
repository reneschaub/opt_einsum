import simulate

#cores per SM
N = 64 
divisors = (1,2,4,8,16,32,64)
bankN = 32

shape = [4,3,9,22]
out = (2,0,1,3)


#causes huge write sm bank collisions, repeats every 4 writes so 32/4 = 8x slowdown
shape = [24,24,24,22]
out = (2,0,1,3)

shape = [24,24,24,22]
out = (2,0,1,3)



a,b, c = simulate.rwblock(shape, out, N, divisors) 

##now pad the tensor; that means need to pass the block as well
#simulate.expand_block( a.block, a.padding, a.spillover )
#a,b, c = simulate.rwblock(shape, out, N, divisors, a.block) 

simulate.bank_padding(a, b, c, out, bankN)

simulate.test_bank(N, bankN, a, b, c, out)
aa,bb,cc = a.__dict__, b.__dict__, c.__dict__

h = simulate.arbelo_input( shape, out, a, b )
layers = simulate.arbelo_cascade( h.dim, shape, out, a, b)
simulate.count( layers[-1][0], 0, a )
paths = simulate.precalc( layers, a )
branches = []; address = simulate.struct(); address.it = 0
g = a #!
simulate.reverse_cascade( layers[-1][0], 123, branches, address, a, g )

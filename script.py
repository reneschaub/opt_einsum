import simulate

#cores per SM
N = 64 
divisors = (1,2,4,8,16,32,64)
bankN = 32

shape = [4,3,9,22]
out = (2,0,1,3)


shape = [80,80,80,22]
out = (2,0,1,3)




a,b, c = simulate.rwblock(shape, out, N, divisors) 
#now pad the tensor; that means need to pass the block as well
simulate.expand_block( a.block, a.padding, a.spillover )
a,b, c = simulate.rwblock(shape, out, N, divisors, a.block) 

padded_block = simulate.bank_padding(a, b, c, out, bankN)

simulate.test_bank(N, bankN, a, b, out, padded_block)

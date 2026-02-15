function oc1_leg4(T, AC, Au, Ad; forloop_iter, ifparallel)
    l = FLmap_parallel(T, AC, AC, (Au, Ad); forloop_iter, ifparallel)
    # return sum(ein"abcd,abcd->"(l,T))
    return sum(@tensor l[a,b,c,d] * T[a,b,c,d])
end

function contract_n1(T, AC, A; forloop_iter, ifparallel)
    return oc1_leg4(T, AC, A, conj(A); forloop_iter, ifparallel)
end

function contract_o1(T, AC, A, O; forloop_iter, ifparallel)
    @tensor Au[1,2,3,4,5] := A[1,2,3,4,6] * O[6,5]
    Ad = conj(A)
    return oc1_leg4(T, AC, Au, Ad; forloop_iter, ifparallel)
end

function oc_H_leg4(T, AC, AL, A1u, A1d, A2u, A2d; ifparallel, forloop_iter)
    l = FLmap_parallel(T, AL, AL, (A1u, A1d); ifparallel, forloop_iter)
    l = FLmap_parallel(l, AC, AC, (A2u, A2d); ifparallel, forloop_iter)
    # return sum(ein"abcd,abcd->"(l,T))
    return sum(@tensor l[a,b,c,d] * T[a,b,c,d])
end


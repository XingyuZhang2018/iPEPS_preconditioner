"""
    restriction_ipeps(ipeps)
return a `SquareIPEPS` based on `ipeps` that is symmetric under
permutation of its virtual indices.
```
        4
        │
 1 ── ipeps ── 3
        │
        2
```
"""
function _restriction_ipeps(A)
    A = C4v_restriction(A)
    return A
 end

 function C4v_restriction(A)
    A += permutedims(conj(A), (1,4,3,2,5)) # up-down
    A += permutedims(conj(A), (3,2,1,4,5)) # left-right
    A += permutedims(conj(A), (2,1,4,3,5)) # diagonal
    A += permutedims(conj(A), (4,3,2,1,5)) # rotation

    return A
end
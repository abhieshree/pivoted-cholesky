# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 16:48:44 2020

@author: abhieshree
"""

import numpy as np

#define a function which implements the pivoted cholesky algorithm for positive definite/semidefinite matrices.
#a is the input pd matrix of size n x n and k is the rank(<n) it should be approximated to
def pivoted_cholesky(a, k):
    r=0
    A = np.array(a, float)
    L = np.zeros_like(A)    
    nrows, ncols = np.shape(A)    
    p = np.array(range(0,nrows))
    immutable_d = A.diagonal()
    d = np.array(immutable_d)
    
    #start algorithm: Pivoted cholesky
    while(r<k):
        
        #find pivot: indices of maximum number in diagonal
        imax = r
        
        for j in range(r+1,nrows):
            if(d[p[j]] > d[p[imax]]):
                imax = j
                
        #swap p[r] and p[imax]
        p[r], p[imax] = p[imax], p[r]
        
        #diagonal element
        L[p[r],r] = np.sqrt(d[p[r]])
        
        #other elements
        for i in range(r+1,nrows):
            L[p[i],r] = (A[p[r],p[i]] - sum(L[p[r],:r]*L[p[i],:r]))/L[p[r],r]
            d[p[i]] = d[p[i]] - L[p[i],r]**2
        
        r = r+1
        
    return L[:,:k], p[:k]




#Examples A1=positive definite matrix, A2 = positive semi definite matrix with rank 3
# A1 = np.array([[3,4,3],
#               [4,8,6],
#               [3,6,9]])

# A2 = np.array([[2.51,4.04,3.34,1.34,1.29],
#               [4.04,8.22,7.38,2.68,2.44],
#               [3.34,7.38,7.06,2.24,2.14],
#               [1.34,2.68,2.24,0.96,0.80],
#               [1.29,2.44,2.14,0.80,0.74]])

 
# print("Output for A1:")
# print(pivoted_cholesky(A1, 3))

# print("Output for A2:")
# print(pivoted_cholesky(A2, 3))
    
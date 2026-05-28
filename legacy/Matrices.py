# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:45:32 2019

@author: rfern
"""
#%% Tridiagonal matrices as regular arrays with numpy
import numpy as np


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

a = [1, 1]; b = [2, 2, 2]; c = [3, 3]
A = tridiag(a, b, c)
#%% Tridiagonal matrices as sparse matrices with spicy
from scipy.sparse import diags
diagonals = [[1, 2, 3, 4], [1, 2, 3], [1, 2]]
A=diags(diagonals, [0, -1, 2])
type(A)
print('Only the non-zero entries are kept')
print(A)
print('As an array, all the entries are kept')
print(A.toarray())

#%% Triadiagonal bock matrices as sparse matrices with spicy
from scipy.sparse import coo_matrix, block_diag, hstack, vstack, dia_matrix
# coo_matrix : A sparse matrix in COOrdinate format
# block_diag : Create a block diagonal matrix from provided arrays

A = coo_matrix([[1,2],[3,4]])
B = coo_matrix ([[5],[6]])
C = coo_matrix([[7]])

block_diag((A,B,C)).toarray()
#%% TRIDIAGONAL BLOCK MATRIX FROM 3 LISTS OF COO MATRICES
from scipy.sparse import coo_matrix, block_diag, hstack, vstack, dia_matrix
Diagonal =[coo_matrix([[1,2],[3,4]]),coo_matrix([[1],[3]]),coo_matrix([[1]])]
SubDiagonal=[coo_matrix([[1,2],[3,4]]),coo_matrix([[1]])]
SuperDiagonal=[coo_matrix([[1],[3]]),coo_matrix([[1],[3]])]

def TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal):
    n1=Diagonal[0].shape[0]
    m1=Diagonal[0].shape[1]
    n2=Diagonal[-1].shape[0]
    m2=Diagonal[-1].shape[1]

    Diagonal=block_diag(Diagonal,dtype='float64')
    
    if(len(SubDiagonal)==0 or len(SuperDiagonal)==0):
        return(Diagonal)
        
    SubDiagonal=block_diag(SubDiagonal,dtype='float64')
    Zeros=coo_matrix((n1, SubDiagonal.shape[1]))
    SubDiagonal=vstack([Zeros,SubDiagonal])
    Zeros=coo_matrix((SubDiagonal.shape[0],m2))
    SubDiagonal=hstack([SubDiagonal,Zeros])

    SuperDiagonal=block_diag(SuperDiagonal,dtype='float64')
    Zeros=coo_matrix((n2, SuperDiagonal.shape[1]))
    SuperDiagonal=vstack([SuperDiagonal,Zeros])
    Zeros=coo_matrix((SuperDiagonal.shape[0],m1))
    SuperDiagonal=hstack([Zeros,SuperDiagonal])

    return(SubDiagonal + Diagonal + SuperDiagonal)


def SupDiagBlockMatrix(Diagonal,SuperDiagonal):
    m1=Diagonal[0].shape[1]
    m2=SuperDiagonal[-1].shape[1]
    
    Diagonal=block_diag(Diagonal,dtype='float64')    
    Zeros=coo_matrix((Diagonal.shape[0],m2))
    Diagonal=hstack([Diagonal,Zeros])
    
    SuperDiagonal=block_diag(SuperDiagonal,dtype='float64')
    Zeros=coo_matrix((SuperDiagonal.shape[0],m1))
    SuperDiagonal=hstack([Zeros,SuperDiagonal])
    
    return(Diagonal + SuperDiagonal)

def SubDiagBlockMatrix(SubDiagonal,Diagonal):
    n1=Diagonal[0].shape[0]
    n2=SubDiagonal[-1].shape[0]
    
    Diagonal=block_diag(Diagonal,dtype='float64')    
    Zeros=coo_matrix((n2,Diagonal.shape[1]))
    Diagonal=vstack([Diagonal,Zeros])

    SubDiagonal=block_diag(SubDiagonal,dtype='float64')
    Zeros=coo_matrix((n1,SubDiagonal.shape[1]))
    SubDiagonal=vstack([Zeros,SubDiagonal])
    return(SubDiagonal+Diagonal)









from scipy.sparse import csc_matrix, block_diag, hstack, vstack
#%% TRIDIAGONAL BLOCK MATRIX FROM 3 LISTS OF COO MATRICES
def TriBlockMatrix(SubDiagonal,Diagonal,SuperDiagonal):
    n1=Diagonal[0].shape[0]
    m1=Diagonal[0].shape[1]
    n2=Diagonal[-1].shape[0]
    m2=Diagonal[-1].shape[1]

    Diagonal=block_diag(Diagonal,dtype='float64',format='csc')
    
    if(len(SubDiagonal)==0 or len(SuperDiagonal)==0):
        return(Diagonal)
        
    SubDiagonal=block_diag(SubDiagonal,dtype='float64',format='csc')
    Zeros=csc_matrix((n1, SubDiagonal.shape[1]))
    SubDiagonal=vstack([Zeros,SubDiagonal])
    Zeros=csc_matrix((SubDiagonal.shape[0],m2))
    SubDiagonal=hstack([SubDiagonal,Zeros],format='csc')

    SuperDiagonal=block_diag(SuperDiagonal,dtype='float64',format='csc')
    Zeros=csc_matrix((n2, SuperDiagonal.shape[1]))
    SuperDiagonal=vstack([SuperDiagonal,Zeros])
    Zeros=csc_matrix((SuperDiagonal.shape[0],m1))
    SuperDiagonal=hstack([Zeros,SuperDiagonal],format='csc')

    return(SubDiagonal + Diagonal + SuperDiagonal)

#%% BLOCK MATRIX WITH DIAGONAL AND SUPERDIAGONAL FROM 2 LISTS OF COO MATRICES
def SupDiagBlockMatrix(Diagonal,SuperDiagonal):
    m1=Diagonal[0].shape[1]
    m2=SuperDiagonal[-1].shape[1]
    
    Diagonal=block_diag(Diagonal,dtype='float64',format='csc')    
    Zeros=csc_matrix((Diagonal.shape[0],m2))
    Diagonal=hstack([Diagonal,Zeros],format='csc')
    
    SuperDiagonal=block_diag(SuperDiagonal,dtype='float64',format='csc')
    Zeros=csc_matrix((SuperDiagonal.shape[0],m1))
    SuperDiagonal=hstack([Zeros,SuperDiagonal],format='csc')
    
    return(Diagonal + SuperDiagonal)
#%% BLOCK MATRIX WITH DIAGONAL AND SUBRDIAGONAL FROM 2 LISTS OF COO MATRICES
def SubDiagBlockMatrix(SubDiagonal,Diagonal):
    n1=Diagonal[0].shape[0]
    n2=SubDiagonal[-1].shape[0]
    
    Diagonal=block_diag(Diagonal,dtype='float64',format='csc')    
    Zeros=csc_matrix((n2,Diagonal.shape[1]))
    Diagonal=vstack([Diagonal,Zeros],format='csc')

    SubDiagonal=block_diag(SubDiagonal,dtype='float64',format='csc')
    Zeros=csc_matrix((n1,SubDiagonal.shape[1]))
    SubDiagonal=vstack([Zeros,SubDiagonal],format='csc')
    return(SubDiagonal+Diagonal)
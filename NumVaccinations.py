import InfinitesimalGeneratorVaccinations as IGV
from numpy.linalg import inv
from numpy.linalg import multi_dot
import Parameters as par
N=par.global_N


def FV(q=0.99):
    print('|----- Computing the PDF of the # of vaccinations -----|')
    # Paso 0
    i=1
    r=0
    H=[]
    b=[]
    print('Inverse Step',i)
    H.append(-inv(IGV.Aii_star(i)+IGV.Aii_u(i)))
    b.append(H[i-1].dot(IGV.A1ii_u(i).sum(axis=1)))
    
    while i<N:
        i=i+1
        print('Inverse Step',i)
        H.append(-inv(IGV.Aii_star(i)+IGV.Aii_u(i)+multi_dot([IGV.A1ii_u(i),H[i-2],IGV.Aii1(i-1)])))
        b.append(multi_dot([H[i-1],IGV.A1ii_u(i),b[i-2]]))
    
    # Paso 1
    p_invasion=[]
    F_invasion=[]
    p=[[]]*N
    p[i-1]=b[i-1]
    while i>1:
        i=i-1
        p[i-1]=b[i-1]+multi_dot([H[i-1],IGV.Aii1(i),p[i]])
    p_invasion.append(float(p[0][N-1]))
    F_invasion.append(p_invasion[0])
    
    #Paso 2
    print('Multiplying matrices of max order', H[0].shape,'in every step')
    while F_invasion[r]<=q and r<1000:
        print('Step r',r)
        r=r+1
        c=[]
        if r==1:
            c.append(H[i-1].dot(IGV.A1ii_v(i).sum(axis=1)+IGV.Aii_v(i).dot(p[i-1])))
        else:
            c.append(multi_dot([H[i-1],IGV.Aii_v(i),p[i-1]]))
        while i<N:
            i=i+1
            c.append(H[i-1].dot(IGV.A1ii_u(i).dot(c[i-2])+IGV.A1ii_v(i).dot(p[i-2])+IGV.Aii_v(i).dot(p[i-1])))        
        p[i-1]=c[i-1]
        while i>1:
            i=i-1
            p[i-1]=c[i-1]+multi_dot([H[i-1],IGV.Aii1(i),p[i]])
        p_invasion.append(float(p[0][N-1]))
        F_invasion.append(p_invasion[r]+F_invasion[r-1])
    return F_invasion,p_invasion

def M(K=2):
    i=1
    k=1
    H=[]
    d=[]
    print('Inverse Step',i)
    H.append(-inv(IGV.Aii(i)))
    d.append(H[i-1].dot(IGV.A1ii_v(i).sum(axis=1)+IGV.Aii_v(i).sum(axis=1)))
    while i<N:
        i=i+1
        print('Inverse Step',i)
        H.append(-inv(IGV.Aii(i)+multi_dot([IGV.A1ii(i),H[i-2],IGV.Aii1(i-1)])))
        d.append(H[i-1].dot(IGV.A1ii(i).dot(d[i-2])+IGV.A1ii_v(i).sum(axis=1)+IGV.Aii_v(i).sum(axis=1)))
    M=[[]]*N
    M_invasion=[]
    M[i-1]=d[i-1]
    while i>1:
        i=i-1
        M[i-1]=d[i-1]+multi_dot([H[i-1],IGV.Aii1(i),M[i]])
    M_invasion.append(float(M[0][N-1]))
    while k<K:
        k=k+1
        d[i-1]=k*multi_dot([H[i-1],IGV.Aii_v(i),M[i-1]])
        while i<N:
            i=i+1
            d[i-1]=H[i-1].dot(IGV.A1ii(i).dot(d[i-2])+k*IGV.A1ii_v(i).dot(M[i-2])+k*IGV.Aii_v(i).dot(M[i-1]))
        M[i-1]=d[i-1]
        while i>1:
            i=i-1
            M[i-1]=d[i-1]+multi_dot([H[i-1],IGV.Aii1(i),M[i]])
        M_invasion.append(float(M[0][N-1]))
    return M_invasion
    
    
    
import InfinitesimalGeneratorR0 as IGR
from numpy.linalg import inv
from numpy.linalg import multi_dot
import Parameters as par
N=par.global_N


def FR0(cuantil=0.99):
    # Step 0
    i=1
    r=0
    H=[]
    b=[]
    print('Inverse Step',i)
    H.append(-inv(IGR.Aii(i)))
    b.append(H[i-1].dot(IGR.A1ii_bar(i).sum(axis=1)))
    while i<N:
        i=i+1
        print('Inverse Step',i)
        H.append(-inv(IGR.Aii(i)+multi_dot([IGR.A1ii_tilde(i),H[i-2],IGR.Aii1_tilde(i-1)])))
        b.append(H[i-1].dot(IGR.A1ii_bar(i).sum(axis=1)+IGR.A1ii_tilde(i).dot(b[i-2])))
    q_invasion=[]
    G_invasion=[]
    q=[[]]*N
    q[i-1]=b[i-1]
    while i>1:
        i=i-1
        q[i-1]=b[i-1]+multi_dot([H[i-1],IGR.Aii1_tilde(i),q[i]])
    q_invasion.append(q[0][N-1])
    G_invasion.append(q_invasion[0])
    # Step 2
    while G_invasion[r]<cuantil:
        r=r+1
        c=[]
        c.append(multi_dot([H[i-1],IGR.Aii1_bar(i),q[i]]))
        while i<N:
            i=i+1
            if i<N:
                c.append(H[i-1].dot(IGR.A1ii_tilde(i).dot(c[i-2])+IGR.Aii1_bar(i).dot(q[i])))
            else:
                c.append(multi_dot([H[i-1],IGR.A1ii_tilde(i),c[i-2]]))
        q[i-1]=c[i-1]
        while i>1:
            i=i-1
            q[i-1]=c[i-1]+multi_dot([H[i-1],IGR.Aii1_tilde(i),q[i]])
        q_invasion.append(q[0][N-1])
        G_invasion.append(q_invasion[r]+G_invasion[r-1])
    return G_invasion,q_invasion

def M(K=1):
    i=1
    k=1
    H=[]
    d=[]
    print('Inverse Step',i)
    H.append(-inv(IGR.Aii(i)))
    d.append(H[i-1].dot(IGR.Aii1_bar(i).sum(axis=1)))
    while i<N:
        i=i+1
        print('Inverse Step',i)
        H.append(-inv(IGR.Aii(i)+multi_dot([IGR.A1ii_tilde(i),H[i-2],IGR.Aii1(i-1)])))
        if i<N:
            d.append(H[i-1].dot(IGR.A1ii_tilde(i).dot(d[i-2])+k*IGR.Aii1_bar(i).sum(axis=1)))
        else:
            d.append(multi_dot([H[i-1],IGR.A1ii_tilde(i),d[i-2]]))
    M_invasion=[]
    M=[[]]*N
    M[i-1]=d[i-1]
    while i>1:
        i=i-1
        M[i-1]=d[i-1]+multi_dot([H[i-1],IGR.Aii1(i),M[i]])
    M_invasion.append(M[0][N-1])
    while k<K:
        k=k+1
        d[i-1]=k*multi_dot([H[i-1],IGR.Aii1_bar(i),M[i]])
        while i<N:
            i=i+1
            if i<N:
                d[i-1]=H[i-1].dot(IGR.A1ii_tilde(i).dot(d[i-2])+k*IGR.Aii1_bar(i).dot(M[i]))
            else:
                d[i-1]=multi_dot([H[i-1],IGR.A1ii_tilde(i),d[i-2]])
        M[i-1]=d[i-1]
        while i>1:
            i=i-1
            M[i-1]=d[i-1]+multi_dot([H[i-1],IGR.Aii1(i),M[i]])
        M_invasion.append(M[0][N-1])
    return M[0][0:N]

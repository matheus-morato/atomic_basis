import sys
import numpy as np
from math import factorial
from scipy.special import factorial2

orbital_list={ 's' : int(0),
               'p' : int(1),
               'd' : int(2),
               'f' : int(3),
               'g' : int(4),
               'h' : int(5),
               'i' : int(6),
               'j' : int(7)}


def open_basis(basis_file):
    '''Reads and parses a basis set file in MOLPRO's output format.
       That is,
          #[expoent]   [1th contraction] ... [i-th contraction] ...
       -------  Example  -----------
       10.0         0.5  -0.2  0.0
        2.0         0.5   0.8  0.0
        0.5         0.0   0.1  1.0      
       -----------------------------

       The basis set does not need to be normalized, but needs to be orthogonal.


       #TODO: ignore comment lines.
    '''
    bas=open(basis_file,'r')
    exp=[]
    coef=[]
    inter=0
    for i in bas:
        sep=i.split()
        if inter == 0:
            for l in range(len(sep)-1):
                coef.append([])
            inter=1
        count=1
        for a in sep:
            if count == 1:
                exp.append(float(a))
                count+=1
            else:
                coef[count-2].append(float(a))
                count+=1
    Contraction=np.matrix(coef)
    bas.close()
    return exp,Contraction

def orb_from_file(coef_file,mult=False):
    '''Reads a file containing atomic orbitals coefficients.
       The number of coefficients must be the same as the number of contracted basis functions.
    '''
    co=open(coef_file,'r')
    if mult:
        Coef=[]
    for i in co:
        i_mod=[]
        for a in i.split():
            i_mod.append(float(a))
        if mult:
            Coef.append(np.matrix(i_mod))
        else:
            Coef=np.matrix(i_mod)
    co.close()
    return Coef

def contraction_normalizer(contr,exp,sym,dont=False,debug=False):
    ''' Normalize the contracted functions.'''
    if dont:
        if debug: print('Contraction normalization: off')
        return contr
    if debug: print('Full contraction:\n',contr,'\n') 
    #if debug: print('Coefficients:\n',coef,'\n')
    max_prim=len(np.ravel(contr[0]))
    max_contr=len(contr)
    if debug: print('Number of primitive functions: ',max_prim,'\n')
    if debug: print('Number of contracted functions: ',max_contr,'\n')
    if max_prim < max_contr: raise ValueError('More contracted ('+str(max_contr)+') than primitive ('+str(max_prim)+') basis set funtion!')
    norm_prim=np.zeros((max_contr,max_prim))
    if debug: print('New normalized primitive:\n',norm_prim,'\n')
    for i,C in enumerate(contr):
        if debug: print('Contraction #'+str(i+1)+':') 
        if debug: print(C) 
        S=0
        for j in range(max_prim):
            for k in range(max_prim):
                S+=C[0,j]*C[0,k]*(4*exp[j]*exp[k]/(exp[j]+exp[k])**2)**((2*sym+3)/4)
        if debug: print('S value:',S) 
        if debug: print('Re-contraction #'+str(i+1)+':') 
        if debug: print(C/np.sqrt(S),'\n')    
        contr[i]=C/np.sqrt(S)
    if debug: print('New contraction matrxi:\n',contr,'\n')
    return contr


def calculator(typ,more,exp,Contraction,Coef):
    '''Calculates the expected radii and radii squared of an orbital.
       In the primitive basis function basis the general formula are:

       <i|r|j> = [(e_i*e_j)^((2l+3)/4)/(e_i+e_j)^(l+2)]*[(l+1)!2^((4l+5)/2)/(sqrt(pi)*(2l+1)!!)]
       and
       <i|r^2|j> = [(4e_i*e_j)^((2l+3)/4)/(e_i+e_j)^((2l+5)/2)]*(2l+3)/2

       where e_i is the expoent of |i> and l is the orbital angular number (0 for s, 1 for p, ...).
    '''
    if typ == 'r':
        tam=len(exp)
        Integrals=np.zeros((tam,tam))                          
        if not isinstance(more, int) or more < 0:
            raise ValueError('The orbital angular moment must be a natural number.')
        fac_value=factorial(more+1)/factorial2(2*more+1, exact=True) 
        print(fac_value)
        for a in range(tam):                                   
            for b in range(tam):
                Integrals[a][b]=((exp[a]*exp[b])**((2*more+3)/4)/(exp[a]+exp[b])**(more+2))*(2**((4*more+5)/2)*fac_value/(np.pi**(1/2)))
    elif typ == 'r2':
        tam=len(exp)
        Integrals=np.zeros((tam,tam))
        for a in range(tam):
            for b in range(tam):
                Integrals[a][b]=((4*exp[a]*exp[b])**((2*more+3)/4)/(exp[a]+exp[b])**((2*more+5)/2))*((2*more+3)/2)

    #print(Contraction)
    #print(np.matmul(np.matmul(Coef,Contraction),Integrals,np.matmul(Contraction.transpose(),Coef.transpose())))
    left=np.matmul(Coef,Contraction)
    #print(left)
    #print(Integrals)
    right=np.matmul(Contraction.transpose(),Coef.transpose())
    #print(right)
    left_int=np.matmul(left,Integrals)#,right))
    #print(np.matmul(left_int,right))
    int_final=np.matmul(left_int,right)
    
    Norm=orbital_normalizer(exp,tam,more)

    left_norm=np.matmul(left,Norm)
    norm_final=np.matmul(left_norm,right)
    return int_final[0][0],norm_final,int_final[0][0]/norm_final

def orbital_normalizer(exp,tam,more):
    '''Normalize the orbitals using the normalization in the primitive basis.
       That is,
       <i|j> = [(4e_i*e_j)/(e_i+e_j)^2]^((2l+3)/4)

       where e_i is the expoent of |i> and l is the orbital angular number (0 for s, 1 for p, ...).
    '''
    tam=len(exp)
    Norm=np.zeros((tam,tam))
    if not isinstance(more, int) or more < 0:
        raise ValueError('The orbital angular moment must be a natural number.')
    for a in range(tam):
        for b in range(tam):
             Norm[a][b]=((4*exp[a]*exp[b])/(exp[a]+exp[b])**2)**((2*more+3)/4) 
     
    return Norm

if __name__ == '__main__':
    typ=sys.argv[1]
    basis_file=sys.argv[2]
    coef_file=sys.argv[3]
    more=orbital_list[sys.argv[4]]
    exp,Contraction=open_basis(basis_file)
    Coef=orb_from_file(coef_file,mult=True)
    norm_contraction=contraction_normalizer(Contraction,exp,more,dont=False,debug=False)
    #norm_contraction=contraction_normalizer(Contraction,exp,more,dont=False,debug=True)
    #result=calculator(typ,more,exp,Contraction,Coef)
    if typ == 'full':
        for i in range(len(Coef)):
            print('Orbital #{0}'.format(i))
            result=calculator('r',more,exp,norm_contraction,Coef[i])
            print('Norm = ',result[1])
            print('r-Value = ',result[0])
            print('r-Value/Norm = ',result[2])
            result2=calculator('r2',more,exp,norm_contraction,Coef[i])
            print('r2-Value = ',result2[0])
            print('r2-Value/Norm = ',result2[2])
            print('r/sqrt(r2) = ',result[2]/np.sqrt(result2[2]))
    else:
        for i in range(len(Coef)):
            print('Orbital #{0}'.format(i))
            result=calculator('r',more,exp,norm_contraction,Coef[i])
            print('Value = ',result[0])
            print('Norm = ',result[1])
            print('Value/Norm = ',result[2])

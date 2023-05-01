import sys
import numpy as np
import r_r2 as r_cal


'''Calculate the projection of a first set of vectors onto the space spawn by a second set of vectors.
   It assumes that each set is linearly independent.

   TODO: Orthonormalize the vectors if the set is linearly dependent.
'''

basis_file=sys.argv[1]
first_coef_file=sys.argv[2] 
second_coef_file=sys.argv[3] 
more=r_cal.orbital_list[sys.argv[4]]

exp,Contraction=r_cal.open_basis(basis_file)
tam=len(exp)
norm_contraction=r_cal.contraction_normalizer(Contraction,exp,more,dont=False,debug=False)


Coef_1=r_cal.orb_from_file(first_coef_file,mult=True)
Coef_2=r_cal.orb_from_file(second_coef_file,mult=True)
Norm=r_cal.orbital_normalizer(exp,tam,more)
#print(Norm)

int_final=np.zeros((len(Coef_1)+1,len(Coef_2)+1))
#print(len(Coef_1),len(Coef_2))

for n1,c1 in enumerate(Coef_1):
    left=np.matmul(c1,norm_contraction)
    left_int=np.matmul(left,Norm)
    left_norm=np.matmul(left_int,left.transpose())
    #print(left_norm)
    for n2,c2 in enumerate(Coef_2):
        right=np.matmul(norm_contraction.transpose(),c2.transpose())
        right_int=np.matmul(Norm,right)
        right_norm=np.matmul(right.transpose(),right_int)
        #print(right_norm)
        val=np.matmul(left_int,right)/(np.sqrt(right_norm*left_norm))
        int_final[n1,n2]=val
        int_final[n1,-1]+=val**2
        int_final[-1,n2]+=val**2

for n1 in range(len(Coef_1)):
    int_final[n1][-1]=np.sqrt(int_final[n1][-1])
    print('<C_2|{0}d> = {1:0.3f}'.format(n1+3,int_final[n1][-1]))

for n2 in range(len(Coef_2)):
    int_final[-1][n2]=np.sqrt(int_final[-1][n2])
    print('<C_1|{0}d> = {1:0.3f}'.format(n2+3,int_final[-1][n2]))

print(int_final)
#print(Coef_1)
#print(Coef_2)



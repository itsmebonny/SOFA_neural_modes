import time
import numpy as np
import torch
from einops import einsum, rearrange
from torch import Tensor 
from typing import Dict

# this indices matrix helps the computation of cofactor matrices in 3d
_minorIdx3d = torch.empty((3, 3, 2, 2, 2), dtype=int)
for i in torch.arange(3):
    for j in torch.arange(3):
        posi, posj = 0, 0
        for si in torch.arange(3):
            for sj in torch.arange(3):
                if si != i and sj != j:
                    _minorIdx3d[i, j, posi, posj, 0] = si
                    _minorIdx3d[i, j, posi, posj, 1] = sj
                    posj += 1
                    if posj == 2:
                        posi = 1
                        posj = 0
_minorSigns3d = torch.ones((3, 3))
_minorSigns3d[0, 1] = -1
_minorSigns3d[1, 0] = -1
_minorSigns3d[1, 2] = -1
_minorSigns3d[2, 1] = -1

# this indices matrix helps the computation of cofactor matrices in 2d
_minorIdx2d = torch.asarray([
   [[1, 1], [1, 0]],
   [[0, 1], [0, 0]]
])
_minorSigns2d = torch.ones((2, 2))
_minorSigns2d[0, 1] = -1
_minorSigns2d[1, 0] = -1

'''Converts 2d gradient and hessian matrices to 3d'''
def _convertTo3D(gradU, hessU=None):
    dimension = gradU.shape[-1]

    # convert 2d gradient and hessian to 3d
    # using plane strain assumptions (du/dz = 0)
    if dimension == 2:
        dimension = 3
        gradU2d, hessU2d = gradU, hessU
        gradU = torch.zeros((gradU.shape[0], dimension, dimension), device=gradU.device)
        gradU[..., :2, :2] = gradU2d

        if hessU2d is not None:
            hessU = torch.zeros((hessU.shape[0], dimension, dimension, dimension), device=hessU.device)
            hessU[..., :2, :2, :2] = hessU2d

    return gradU, hessU

'''Computes the Cauchy stress tensor of an isotropic, homogeneous, linearly elastic material'''
def linearSigma(gradU, mu, lmbd):
    # reshape so that gradU is always N x dim x dim
    if gradU.ndim != 3: gradU = gradU[None, ...]

    # convert gradients to 3D if necessary
    dimension = 3
    gradU, _ = _convertTo3D(gradU)

    # the second-order identity tensor
    I = torch.eye(dimension, device=gradU.device)

    return lmbd * torch.einsum('pkk,ij->pij', gradU, I) \
        + mu * (gradU + torch.einsum('pij->pji', gradU))

'''Computes the divergence of the linear elasticity stress tensor'''
def divLinearSigma(gradU, hessU, mu, lmbd):
    return (mu + lmbd) * torch.einsum('...jij->...i', hessU) + mu * torch.einsum('...ijj->...i', hessU)

'''Computes the first Piola-Kirchhoff stress tensor of 
a Saint Venant Kirchhoff material'''
def PK1SaintVenantKirchhoff(gradU, mu, lmbd):
    # reshape so that gradU is always N x dim x dim
    if gradU.ndim != 3: gradU = gradU[None, ...]

    # convert gradients to 3D if necessary
    gradU, _ = _convertTo3D(gradU)

    # compute gradient of deformation
    dimension = 3
    I = torch.eye(dimension, device=gradU.device)
    F = gradU + I

    # compute the trace of C minus the trace of I
    trC_trI = torch.einsum('...mn,...mn->...', F, F) - dimension

    # the second Piola-Kirchhoff stress tensor
    S = 0.5 * lmbd * torch.einsum('...,...ij->...ij', trC_trI, I) \
        + mu * (torch.einsum('...mi,...mj', F, F) - I)
    
    # P = FS, the first Piola-Kirchhoff tensor
    return torch.einsum('...im,...mj->...ij', F, S)

'''Computes the divergence of the first Piola-Kirchhoff tensor of 
a Saint Venant Kirchhoff material'''
def divPK1SaintVenantKirchhoff(gradU, hessU, mu, lmbd):
    # reshape so that gradU is always N x dim x dim
    if gradU.ndim != 3: gradU = gradU[None, ...]

    # convert gradients to 3D if necessary
    gradU, hessU = _convertTo3D(gradU, hessU)

    # compute gradient of deformation
    dimension = 3
    I = torch.eye(dimension, device=gradU.device)
    F = gradU + I

    # compute the trace of C minus the trace of I
    trC_trI = torch.einsum('...mn,...mn->...', F, F) - dimension

    # the second Piola-Kirchhoff stress tensor
    S = 0.5 * lmbd * torch.einsum('...,...ij->...ij', trC_trI, I) \
        + mu * (torch.einsum('...mi,...mj', F, F) - I)
    
    # the divergence of the second Piola-Kirchhoff tensor
    divS = lmbd * (
            torch.einsum('...min,...mn->...i', hessU, gradU) +
            torch.einsum('...mim->...i', hessU)
        ) + mu * (
            torch.einsum('...mj,...mij->...i', F, hessU) +
            torch.einsum('...mi,...mjj->...i', F, hessU)
        )
    
    # compute the divergence of P = FS, the first Piola-Kirchhoff tensor
    return torch.einsum('...imj,...mj->...i', hessU, S) + torch.einsum('...im,...m->...i', F, divS)

'''Computes the first Piola-Kirchhoff stress tensor of 
a compressible neohookean material'''
def PK1Neohook(gradU, mu, lmbd):
    # reshape so that gradU is always N x dim x dim
    if gradU.ndim != 3: gradU = gradU[None, ...]

    # convert gradients to 3D if necessary
    gradU, _ = _convertTo3D(gradU)

    # compute gradient of deformation
    dimension = 3
    F = gradU + torch.eye(dimension, device=gradU.device)

    # and its inverse
    inv_F = torch.linalg.inv(F)
    
    # determinant of F
    J = torch.linalg.det(F)

    # the first Piola-Kirchhoff stress tensor
    return mu * (F - torch.einsum('pij->pji', inv_F)) + 0.5 * lmbd * torch.einsum('p,pij->pji', (J ** 2 - 1), inv_F)

'''Computes the divergence of the first Piola-Kirchhoff tensor of 
a compressible neohookean material'''
def divPK1Neohook(gradU, hessU, mu, lmbd): 
    # transfer idx matrices to device
    minorIdx2d = _minorIdx2d.to(gradU.device)
    minorIdx3d = _minorIdx3d.to(gradU.device)
    minorSigns2d = _minorSigns2d.to(gradU.device)
    minorSigns3d = _minorSigns3d.to(gradU.device)

    # reshape so that gradU is always N x dim x dim
    # and hessU is always N x dim x dim x dim
    if gradU.ndim < 3: gradU = gradU[None, ...]
    if hessU.ndim < 4: hessU = hessU[None, ...]

    # convert gradients to 3D if necessary
    dimension = 3
    gradU, hessU = _convertTo3D(gradU, hessU)
    # print('dimension', gradU.shape)
    # print('dimension', dimension)

    # compute gradient of deformation
    F = gradU + torch.eye(dimension, device=gradU.device)
    
    # indexed gradient of deformation [F]i',j'
    # this is the squared matrix with the values of F that would yield to the minors matrix of F
    # if the determinant of this matrix is computed over the last two dimensions
    idxF = F[..., minorIdx3d[..., 0], minorIdx3d[..., 1]]

    # indexed gradient of F (hessian of U)
    # spatial derivatives of gradient of deformation F, indexed as above for F
    # this matrix allows the computation of spatial derivatives with respect to 
    # cofactors i,j
    idxHessU = torch.empty((*hessU.shape[0:-3], dimension, dimension,  2, 2, dimension), device=hessU.device)
    for jdx in range(dimension):
        idxHessU[..., jdx] = hessU[..., minorIdx3d[..., 0], minorIdx3d[..., 1], jdx]
    # print('idxHessU computed ...')
    
    # now compute the cofactor matrices of each squared submatrix from the indexed hessian matrix
    cofIdxHessU = torch.einsum('...pqj,pq->...pqj', idxHessU[..., minorIdx2d[..., 0], minorIdx2d[..., 1], :], minorSigns2d)
    # print('cofIdxHessU computed ...')
    
    # matrix of cofactors of F
    # compute the determinant of the squared submatrices
    # of the last two dimensions of the indexed F matrix
    # and multiply by minor signs accordingly
    cofF = torch.einsum('...ij,ij->...ij', torch.linalg.det(idxF), minorSigns3d)
    # print('cofF computed ...')
    # the determinant can be computed using any index:
    # det(F) = <F_m1, C_m1> = <F_m2, C_m2> = <F_m3, C_m3>
    # hence its gradient can also be computed using any index.
    # here I use i=1 for both
    i = 0

    # determinant of F and its inverse
    J = torch.einsum('...m,...m', F[..., i], cofF[..., i]) + 1e-8
    invJ = 1 / J
    # print('J and 1 / J computed ...')

    # gradient of cofactor matrix of F
    # NOTE: minorSigns3d = minorSigns3D^T
    gradCofF = torch.einsum('mi,...mipq,...mipqj->...mij', minorSigns3d, idxF, cofIdxHessU)
    # print('gradCofF computed ...')

    # gradient of determinant of F
    gradJ = torch.einsum('...m,...mj->...j', cofF[..., i], hessU[..., i, :])
    gradJ += torch.einsum('...mj,...m->...j', gradCofF[..., i, :], F[..., i])
    # print('gradJ computed ...')
    
    # divergence of inverse of F transposed 
    divInvF_T = -torch.einsum('...,...j,...mj->...m', invJ / J, gradJ, cofF) 
    divInvF_T += torch.einsum('...,...mjj->...m', invJ, gradCofF)
    # print('divInvF_T computed ...')

    # divergence of F
    divF = torch.einsum('...mjj->...m', hessU)
    # print('divF computed ...')
    
    # and finally, divergence of PK1
    divPK1 = mu * (divF - divInvF_T)
    if lmbd != 0.0:
        divPK1 += 0.5 * lmbd * torch.einsum('...,...i->...i', (J ** 2 - 1), divInvF_T)
        divPK1 += lmbd * torch.einsum('...ij,...j->...i', cofF, gradJ)

    return divPK1

'''Computes the divergence of the first Piola-Kirchhoff tensor of 
an incompressible neohookean material'''
def divPK1NeohookIncompressible(gradU, hessU, mu):
    # reshape so that gradU is always N x dim x dim
    # and hessU is always N x dim x dim x dim
    if gradU.ndim != 3: gradU = gradU[None, ...]
    if hessU.ndim != 4: hessU = hessU[None, ...]

    # convert gradients to 3D if necessary
    _, hessU = _convertTo3D(gradU, hessU)

    # compute divergence of F
    divF = torch.einsum('...ijj->...i', hessU)
    return mu * divF

'''Computes the strain energy density of a compressible neohookean material.
The bulk term is ogden-type [Holzapfel2000]. The neohook term Eq. (6.147) and bulk term from (6.138).
Also see [Bischoff2001], coupled strain energy potential in Eq. (8), with bulk term 
W2 in Eq. (46) annd [Ciarlet1988, sec. 4.10]'''
def strainEnergyDensityNeohook(gradU, mu, lmbd):
    # reshape so that gradU is always N x dim x dim
    if gradU.ndim < 3: gradU = gradU[None, ...]

    # convert gradients to 3D if necessary
    dimension = 3
    gradU, _ = _convertTo3D(gradU)

    # compute gradient of deformation
    F = gradU + torch.eye(dimension, device=gradU.device)
    
    # determinant of F
    J = torch.linalg.det(F)

    # first invariant of right green lagrange deformation tensor
    IC = torch.einsum('...ji,...ij->...', F, F)

    # strain energy density
    return 0.5 * mu * (IC - 3 - 2 * torch.log(J)) \
        + 0.25 * lmbd * (J ** 2 - 1 - 2 * torch.log(J))



def getGradU3D(field, spacing):
    '''Computes the gradient of a 3D vector field of shape d i j k b, with d components dim, ijk spatial dims 
    and b batch dim. Output b, ijk, component dim (ui,j,k) and spatial deriv (di,j,k).'''
    gradU = torch.stack(torch.gradient(field, spacing=spacing, dim=[1, 2, 3]))
    # put components dim first (ui,j,k) and then spatial (di,j,k)
    gradU = rearrange(gradU, 'd1 d2 i j k b -> b i j k d2 d1')
    return gradU


def getHessU3D(gradU, spacing):
    '''Computes the hessian of a 3D gradient of a vector field of shape b i j k d2 d1, with b, batch dim, ijk spatial dims, 
    d2 component dim (ui,j,k) and d1 spatial deriv (di,j,k). Output b, ijk, d2, d1, d3 second spatial deriv'''
    return torch.stack(torch.gradient(gradU, spacing=spacing, dim=[1, 2, 3]), dim=-1)


class strainEnergyDensityNeohookRegLoss(torch.nn.Module):
    def __init__(self, pred_key, spacing, mu, lmbd, mask_key=None, *args, **kwargs):
        super(strainEnergyDensityNeohookRegLoss, self).__init__(*args, **kwargs)
        self.lmbd = lmbd
        self.mu = mu
        self.pred_key = pred_key
        self.spacing = spacing
        self.mask_key = mask_key

    def forward(self, sample: Dict[str, Tensor]) -> Tensor:
        E_pred = self.computeEnergyDensityNeohook(sample[self.pred_key], sample[self.mask_key])
        summask = np.prod(sample[self.pred_key].shape[:1] + sample[self.pred_key].shape[2:]).item() / max(sample[self.mask_key].sum(), 1)
        return (E_pred ** 2).mean() * summask
    
    def computeEnergyDensityNeohook(self, field, mask):
        # put spatial dims first
        U = rearrange(field, 'b d i j k -> d i j k b')
        # estimate gradient and hessian with finite differences
        gradU = getGradU3D(field, self.spacing)
        # compute Energy
        r = einsum(strainEnergyDensityNeohook(gradU, self.mu, self.lmbd), 
                   mask,
                   'b i j k, b i j k -> b i j k')
        # remove nan
        r[r != r] = 0
        return r
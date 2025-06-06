import os
import time
import torch
import numpy as np
import logging
from abc import ABC, abstractmethod
import pyvista
from dolfinx import plot
import traceback
from dolfinx.io import gmshio
from mpi4py import MPI
import matplotlib.pyplot as plt

from torch import nn
from torch.autograd import Function

# Abstract base classes for modular desig, help me move boundary condition
class EnergyModel(ABC):
    """Abstract energy model defining material behavior"""
    @abstractmethod
    def compute_energy(self, displacement_batch):
        pass
    
    @abstractmethod
    def compute_gradient(self, displacement_batch):
        """Returns internal forces"""
        pass

class LinearSolver(ABC):
    """Abstract solver for linear systems (e.g., CG, direct)"""
    @abstractmethod
    def solve(self, matrix_operator, rhs_batch, **kwargs):
        pass

class Preconditioner(ABC):
    """Abstract preconditioner interface"""
    @abstractmethod
    def compute(self, displacement_batch):
        pass
    
    @abstractmethod
    def apply(self, residual_batch):
        pass



class UFLNeoHookeanModel(torch.nn.Module):
    """
    Modular Neo-Hookean energy model (UFL-Equivalent Formulation).

    Implements the Neo-Hookean formulation:
    W = 0.5 * μ * (I_C - 3 - 2 * ln(J)) + 0.25 * λ * (J² - 1 - 2 * ln(J))
    where:
        μ = Shear modulus
        λ = First Lamé parameter
        F = Deformation gradient
        J = det(F)
        C = Fᵀ F (Right Cauchy-Green tensor)
        I_C = tr(C) (First invariant of C)
    """

    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, dtype=torch.float64):
        """
        Initialize with DOLFINx domain

        Args:
            domain: DOLFINx domain (used for mesh info extraction)
            degree: FEM degree (influences nodes_per_element if not explicit)
            E: Young's modulus
            nu: Poisson's ratio
            precompute_matrices: Whether to precompute FEM matrices (highly recommended)
            device: Computation device (e.g., 'cpu', 'cuda:0')
            dtype: Data type for computation (e.g., torch.float32, torch.float64)
        """
        super(UFLNeoHookeanModel, self).__init__()

        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"Using device: {self.device}, dtype: {self.dtype}")

        # Material properties
        self.E = torch.tensor(E, dtype=self.dtype, device=self.device)
        self.nu = torch.tensor(nu, dtype=self.dtype, device=self.device)
        # Prevent division by zero or instability for nu close to 0.5
        if torch.abs(1.0 - 2.0 * self.nu) < 1e-9:
             print("Warning: nu is close to 0.5. Using a large value for lambda.")
             # Adjust lambda calculation or handle appropriately
             # For practical purposes, maybe cap nu slightly below 0.5
             safe_nu = torch.min(self.nu, torch.tensor(0.49999, dtype=self.dtype, device=self.device))
             self.lmbda = self.E * safe_nu / ((1 + safe_nu) * (1 - 2 * safe_nu))
        else:
             self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) # First Lamé parameter

        self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus

        print(f"Material Properties: E={self.E.item():.2f}, nu={self.nu.item():.2f}, mu={self.mu.item():.2f}, lambda={self.lmbda.item():.2f}")

        # Store domain info (or extracted data)
        self.domain = domain # Keep for potential future use, but extract data now
        self.degree = degree # May influence element type/nodes

        # Extract mesh information from domain
        self._extract_mesh_data() # Needs dolfinx

        # Create element data structure
        self._setup_elements(precompute_matrices)

        # Save configuration parameters
        self.precompute_matrices = precompute_matrices

        # Small value for safe log/sqrt
        self.eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)


    def _extract_mesh_data(self):
        """Extract mesh data from DOLFINx domain"""
        # This part still requires dolfinx to be installed and run once
        try:
            import dolfinx
            import numpy as np
            from dolfinx.fem import FunctionSpace
            from dolfinx.mesh import CellType
        except ImportError:
            raise ImportError("dolfinx is required for mesh data extraction during initialization.")

        # Get coordinates as tensor
        x = self.domain.geometry.x
        self.coordinates = torch.tensor(x, device=self.device, dtype=self.dtype)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        if self.dim != 3:
            print(f"Warning: Expected 3D coordinates, but got {self.dim}D.")

        # --- Determine element type and nodes per element ---
        # Create a dummy function space to query element info
        try:
            # Try vector space with updated API
            element_family = "Lagrange"  # Use Lagrange instead of CG for newer DOLFINx
            element = dolfinx.fem.VectorElement(element_family, self.domain.ufl_cell(), self.degree, self.dim)
            V_dummy = dolfinx.fem.FunctionSpace(self.domain, element)
            dolfin_element = V_dummy.ufl_element()
            self.nodes_per_element = dolfin_element.space_dimension() // self.dim
            cell_type_str = dolfin_element.cell().cellname()

        except Exception as e_vec:
            try:
                # Fallback to scalar space if vector fails
                print(f"Vector space query failed ({e_vec}), trying scalar space...")
                element = dolfinx.fem.FiniteElement(element_family, self.domain.ufl_cell(), self.degree)
                V_dummy = dolfinx.fem.FunctionSpace(self.domain, element)
                dolfin_element = V_dummy.ufl_element()
                self.nodes_per_element = dolfin_element.space_dimension()
                cell_type_str = dolfin_element.cell().cellname()
            except Exception as e_scalar:
                # If both approaches fail, try direct inference based on degree and element type
                print(f"Failed to create function spaces: Vector: {e_vec}, Scalar: {e_scalar}")
                print("Attempting to infer nodes per element from cell type and degree...")
                
                # Fallback to determine nodes_per_element based on common patterns
                cell_type = self.domain.topology.cell_type
                if cell_type == CellType.tetrahedron:
                    cell_type_str = "tetrahedron"
                    # For tetrahedra: degree 1 → 4 nodes, degree 2 → 10 nodes
                    if self.degree == 1:
                        self.nodes_per_element = 4
                    elif self.degree == 2:
                        self.nodes_per_element = 10
                    else:
                        raise ValueError(f"Unsupported degree {self.degree} for tetrahedron")
                elif cell_type == CellType.hexahedron:
                    cell_type_str = "hexahedron"
                    # For hexahedra: degree 1 → 8 nodes, degree 2 → 27 nodes
                    if self.degree == 1:
                        self.nodes_per_element = 8
                    elif self.degree == 2:
                        self.nodes_per_element = 27
                    else:
                        raise ValueError(f"Unsupported degree {self.degree} for hexahedron")
                else:
                    raise ValueError(f"Unsupported cell type: {cell_type}")

        print(f"Detected cell type: {cell_type_str}, Degree: {self.degree}")

        # Continue with the rest of the method...
        # Extract element connectivity
        tdim = self.domain.topology.dim
        
        # Extract connectivity from topology directly if function space approach fails
        try:
            if 'V_dummy' in locals():
                elements_list = V_dummy.dofmap.list.array.reshape(-1, self.nodes_per_element)
            else:
                # Direct topology extraction
                elements_list = []
                for cell in range(self.domain.topology.index_map(tdim).size_local):
                    elements_list.append(self.domain.topology.connectivity(tdim, 0).links(cell))
        except Exception as e_conn:
            print(f"Error extracting element connectivity: {e_conn}")
            print("Attempting manual connectivity extraction...")
            elements_list = []
            for cell in range(self.domain.topology.index_map(tdim).size_local):
                try:
                    # Try different methods to get connectivity
                    conn = self.domain.topology.connectivity(tdim, 0)
                    if hasattr(conn, 'links'):
                        elements_list.append(conn.links(cell))
                    elif hasattr(conn, 'array'):
                        # Some versions use a different API
                        cell_nodes = conn.array[cell * self.nodes_per_element:(cell + 1) * self.nodes_per_element]
                        elements_list.append(cell_nodes)
                    else:
                        raise ValueError("Cannot determine connectivity method")
                except Exception as e_cell:
                    print(f"Failed to extract connectivity for cell {cell}: {e_cell}")
                    # Use placeholder connectivity for robustness
                    elements_list.append(np.arange(self.nodes_per_element, dtype=np.int32))

        self.elements = torch.tensor(np.array(elements_list), device=self.device, dtype=torch.long)
        self.num_elements = len(self.elements)

        print(f"Mesh info extracted: {self.num_nodes} nodes, {self.num_elements} elements")
        print(f"Nodes per element: {self.nodes_per_element} (degree={self.degree}, cell='{cell_type_str}')")

    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        self._generate_quadrature() # Depends on self.nodes_per_element

        if precompute:
            print("Precomputing element shape function derivatives...")
            self._precompute_derivatives()
            self.precomputed = True
            print("Precomputation finished.")
        else:
            print("Derivatives will be computed on-the-fly (less efficient).")
            self.precomputed = False
            # Initialize placeholders if needed later, or handle errors
            self.dN_dx_all = None
            self.detJ_all = None

    def _generate_quadrature(self):
        """Generate quadrature rules based on element type and degree"""
        # Basic quadrature rules - may need refinement for higher accuracy/order
        if self.nodes_per_element == 4:  # Linear Tetrahedron
            # 1-point rule (exact for constant, often sufficient for linear tet volume)
            # self.quadrature_points = torch.tensor([[0.25, 0.25, 0.25]], dtype=self.dtype, device=self.device)
            # self.quadrature_weights = torch.tensor([1.0], dtype=self.dtype, device=self.device) / 6.0
            # 4-point rule (degree 3 exactness) - Often preferred for accuracy
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                                  dtype=self.dtype,
                                                  device=self.device) / 6.0 # Volume of reference tet = 1/6
            print(f"Using 4-point quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 8: # Linear Hexahedron
            # 2x2x2 Gaussian quadrature (8 points)
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
                [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
            ], dtype=self.dtype, device=self.device)
            # Weights for [-1, 1] interval are 1.0 for 2-point Gauss. Volume is 8.
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
            print(f"Using 8-point (2x2x2 Gauss) quadrature for {self.nodes_per_element}-node elements.")
        else:
            # Fallback or error for other element types/degrees
            # For simplicity, using a single point quadrature (centroid) - **likely inaccurate**
            print(f"Warning: Using simple centroid quadrature for {self.nodes_per_element}-node elements. Accuracy may be low.")
            if "tetrahedron" in self._get_cell_type_str(): # Requires _get_cell_type_str helper
                 self.quadrature_points = torch.tensor([[0.25, 0.25, 0.25]], dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.tensor([1.0], dtype=self.dtype, device=self.device) / 6.0
            elif "hexahedron" in self._get_cell_type_str():
                 self.quadrature_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.tensor([8.0], dtype=self.dtype, device=self.device) # Ref volume = 8
            else: # Default guess
                 self.quadrature_points = torch.zeros((1, 3), dtype=self.dtype, device=self.device)
                 self.quadrature_weights = torch.ones((1,), dtype=self.dtype, device=self.device)


        self.num_quad_points = len(self.quadrature_points)

    # Helper to get cell type string robustly after init
    def _get_cell_type_str(self):
         try:
            import dolfinx
            from dolfinx.fem import FunctionSpace
            V_dummy = FunctionSpace(self.domain, ("CG", self.degree))
            return V_dummy.ufl_element().cell().cellname()
         except:
            # Fallback if dolfinx not available later or error
            if self.nodes_per_element == 4: return "tetrahedron"
            if self.nodes_per_element == 8: return "hexahedron"
            return "unknown"


    def _precompute_derivatives(self):
        """Precompute shape function derivatives w.r.t. physical coords (dN/dx)
           and Jacobian determinants (detJ) for all elements and quadrature points."""
        num_qp = self.num_quad_points

        # Allocate tensors
        # Shape: [num_elements, num_quad_points, nodes_per_element, dim]
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, self.dim),
                                    dtype=self.dtype, device=self.device)
        # Shape: [num_elements, num_quad_points]
        self.detJ_all = torch.zeros((self.num_elements, num_qp),
                                   dtype=self.dtype, device=self.device)

        # Process elements - can be done element-wise or batched if memory allows
        # Simpler element-wise loop for clarity:
        for e_idx in range(self.num_elements):
            element_node_indices = self.elements[e_idx]
            element_coords = self.coordinates[element_node_indices] # Coords of nodes for this element

            for q_idx, qp in enumerate(self.quadrature_points):
                # Compute dN/dξ (derivatives in reference element) and then dN/dx
                dN_dxi = self._shape_function_derivatives_ref(qp) # Shape: [nodes_per_element, dim]
                # Compute Jacobian J = dX/dξ = ∑ (Coords_n * dNn/dξ)
                # J_ij = ∑_n (Coords_ni * dNn/dξ_j)
                J = torch.einsum('ni,nj->ij', element_coords, dN_dxi) # Shape: [dim, dim]

                try:
                    detJ = torch.linalg.det(J)
                    invJ = torch.linalg.inv(J)
                except torch.linalg.LinAlgError as err:
                     print(f"Error computing inv/det(J) for element {e_idx} at QP {q_idx}: {err}")
                     print(f"Jacobian Matrix J:\n{J.cpu().numpy()}")
                     # Handle degenerate element - maybe set derivatives to zero or raise error
                     detJ = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                     # Assign zero derivatives to prevent NaN propagation, but this element won't contribute correctly
                     dN_dx = torch.zeros_like(dN_dxi) # Zeros for dN/dx
                     # Or could try pseudo-inverse? invJ = torch.linalg.pinv(J)

                if detJ <= 0:
                     print(f"Warning: Non-positive Jacobian determinant ({detJ.item():.4e}) for element {e_idx} at QP {q_idx}. Check mesh quality.")
                     # Set to small positive value to avoid issues with weights? Or handle upstream.


                # Compute shape function derivatives w.r.t. physical coordinates:
                # dN/dx = dN/dξ * dξ/dX = dN/dξ * J⁻¹
                # dNn/dx_k = ∑_j (dNn/dξ_j * invJ_jk)
                dN_dx = torch.einsum('nj,jk->nk', dN_dxi, invJ) # Shape: [nodes_per_element, dim]

                # Store results
                self.dN_dx_all[e_idx, q_idx] = dN_dx
                self.detJ_all[e_idx, q_idx] = detJ

            # Optional: Add progress indicator for large meshes
            # if (e_idx + 1) % 1000 == 0:
            #     print(f"Precomputing derivatives: {e_idx + 1}/{self.num_elements} elements processed.")

    def _shape_function_derivatives_ref(self, qp_ref):
        """Compute shape function derivatives w.r.t. reference coordinates (ξ, η, ζ)
           at a given reference quadrature point qp_ref."""
        # qp_ref is a tensor [dim] with coordinates in the reference element

        if self.nodes_per_element == 4: # Linear Tetrahedron (Ref coords: L1, L2, L3, L4 sum to 1)
             # Derivatives are constant for linear tet
             # Using barycentric derivatives: dNi/dξj where ξ = (ξ, η, ζ) mapped from barycentric
             # dN1/dξ = -1, dN1/dη = -1, dN1/dζ = -1
             # dN2/dξ = 1,  dN2/dη = 0,  dN2/dζ = 0
             # dN3/dξ = 0,  dN3/dη = 1,  dN3/dζ = 0
             # dN4/dξ = 0,  dN4/dη = 0,  dN4/dζ = 1
             dN_dxi = torch.tensor([
                 [-1.0, -1.0, -1.0],
                 [ 1.0,  0.0,  0.0],
                 [ 0.0,  1.0,  0.0],
                 [ 0.0,  0.0,  1.0]
             ], dtype=self.dtype, device=self.device)

        elif self.nodes_per_element == 8: # Linear Hexahedron (Ref coords: ξ, η, ζ in [-1, 1])
            xi, eta, zeta = qp_ref[0], qp_ref[1], qp_ref[2]

            # Precompute terms
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            xim = one - xi
            xip = one + xi
            etam = one - eta
            etap = one + eta
            zetam = one - zeta
            zetap = one + zeta

            # Shape function derivatives at this point (dN/dξ, dN/dη, dN/dζ)
            # Shape: [nodes_per_element=8, dim=3]
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            eighth = 0.125

            # Derivatives w.r.t. xi (d/dξ)
            dN_dxi[0, 0] = -eighth * etam * zetam
            dN_dxi[1, 0] =  eighth * etam * zetam
            dN_dxi[2, 0] =  eighth * etap * zetam
            dN_dxi[3, 0] = -eighth * etap * zetam
            dN_dxi[4, 0] = -eighth * etam * zetap
            dN_dxi[5, 0] =  eighth * etam * zetap
            dN_dxi[6, 0] =  eighth * etap * zetap
            dN_dxi[7, 0] = -eighth * etap * zetap

            # Derivatives w.r.t. eta (d/dη)
            dN_dxi[0, 1] = -eighth * xim * zetam
            dN_dxi[1, 1] = -eighth * xip * zetam
            dN_dxi[2, 1] =  eighth * xip * zetam
            dN_dxi[3, 1] =  eighth * xim * zetam
            dN_dxi[4, 1] = -eighth * xim * zetap
            dN_dxi[5, 1] = -eighth * xip * zetap
            dN_dxi[6, 1] =  eighth * xip * zetap
            dN_dxi[7, 1] =  eighth * xim * zetap

            # Derivatives w.r.t. zeta (d/dζ)
            dN_dxi[0, 2] = -eighth * xim * etam
            dN_dxi[1, 2] = -eighth * xip * etam
            dN_dxi[2, 2] = -eighth * xip * etap
            dN_dxi[3, 2] = -eighth * xim * etap
            dN_dxi[4, 2] =  eighth * xim * etam
            dN_dxi[5, 2] =  eighth * xip * etam
            dN_dxi[6, 2] =  eighth * xip * etap
            dN_dxi[7, 2] =  eighth * xim * etap

        else:
            # Needs implementation for other element types (e.g., quadratic tet/hex)
            raise NotImplementedError(f"Shape function derivatives not implemented for {self.nodes_per_element}-node elements.")

        return dN_dxi


    def forward(self, u_tensor):
        """
        PyTorch forward method - computes total energy.

        Args:
            u_tensor: Displacement field [num_nodes*dim] or [batch_size, num_nodes*dim]

        Returns:
            Total strain energy (scalar for single input, tensor [batch_size] for batch)
        """
        return self.compute_energy(u_tensor)

    def compute_energy(self, displacement_batch):
        """
        Compute total elastic energy for displacement field(s).

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Total strain energy per batch sample [batch_size] or scalar
        """
        if not self.precomputed:
             raise RuntimeError("Energy computation requires precomputed matrices. Initialize with precompute_matrices=True.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor to [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized Energy Computation ---
        # Gather displacements for all elements: [batch_size, num_elements, nodes_per_element, dim]
        # Need efficient gathering. self.elements gives indices [num_elements, nodes_per_element]
        # u_reshaped is [batch_size, num_nodes, dim]
        # We can use advanced indexing, but need to handle batch dimension.
        # Option 1: Loop over batch (simple, might be okay if batch_size is small)
        # Option 2: Expand elements indices and gather (more complex, potentially faster)

        # Option 1: Loop over batch (simpler to implement/debug)
        total_energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        for b in range(batch_size):
            u_sample = u_reshaped[b] # [num_nodes, dim]
            element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

            # Initialize energy for this sample
            energy_sample = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            # Loop over quadrature points
            for q_idx in range(self.num_quad_points):
                # Get precomputed data for this quad point:
                dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [num_elements, nodes_per_element, dim]
                detJ_q = self.detJ_all[:, q_idx]         # [num_elements]
                qw_q = self.quadrature_weights[q_idx]    # scalar

                # Compute gradient of displacement for all elements at this qp:
                # grad(u)_ij = ∑_n (u_nj * dNn/dx_i)
                # Input element_disps: [E, N, D]
                # Input dN_dx_q:      [E, N, D]
                # Output grad_u:      [E, D, D]
                grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                # Compute deformation gradient F = I + grad(u)
                # Need identity matrix shaped [E, D, D]
                I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                F = I + grad_u # [num_elements, dim, dim]

                # Compute energy density at this quadrature point for all elements
                # Need F with shape [*, 3, 3] for the density function
                energy_density_q = self._compute_neohookean_energy_density(F) # [num_elements]

                # Add weighted contribution to total energy for this sample
                # dEnergy = W * detJ * quad_weight
                energy_sample += torch.sum(energy_density_q * detJ_q * qw_q)

            total_energy[b] = energy_sample

        # Return single value if input wasn't batched
        if not is_batch:
            total_energy = total_energy.squeeze(0)

        return total_energy


    def _compute_neohookean_energy_density(self, F):
        """
        Compute Neo-Hookean strain energy density (UFL-equivalent formulation).

        Args:
            F: Deformation gradient tensor [*, 3, 3] (batch dimension handled by *)

        Returns:
            Energy density W [*]
        """
        # Ensure F has the correct dimensions (at least 3x3)
        if F.shape[-2:] != (3, 3):
             raise ValueError(f"Input F must have shape [..., 3, 3], but got {F.shape}")

        # Determinant J = det(F)
        J = torch.linalg.det(F)

        # Check for non-positive J (potential element inversion)
        if torch.any(J <= self.eps): # Use small epsilon threshold
             # print(f"Warning: Non-positive Jacobian detected (min J = {torch.min(J).item()}). Clamping J for log.")
             # Clamp J to a small positive value for stability in log
             J_safe = torch.clamp(J, min=self.eps)
        else:
             J_safe = J

        # Logarithm of J
        log_J = torch.log(J_safe)

        # First invariant of Right Cauchy-Green C = FᵀF
        # I_C = tr(C) = ∑_i,j (F_ji * F_ji)
        IC = torch.einsum('...ji,...ji->...', F, F) # Sum over last two dimensions

        # Neo-Hookean energy density (W₂)
        # W = (μ/2) * (I_C - 3 - 2*ln(J)) + (λ/4) * (J² - 1 - 2*ln(J))
        W = 0.5 * self.mu * (IC - 3.0 - 2.0 * log_J) + 0.25 * self.lmbda * (J ** 2 - 1.0 - 2.0 * log_J)
        return W

    # --- Methods for Gradients and Stress ---

    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy w.r.t. displacements).

        Uses torch.autograd for automatic differentiation.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Internal forces (nodal forces) [-dE/du] with the same shape as input.
        """
        # Ensure input requires grad
        if not displacement_batch.requires_grad:
            displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
        elif not torch.is_grad_enabled():
             # If called within torch.no_grad(), autograd won't work.
             # Re-enable grad temporarily for this computation.
             # This might be needed if this function is called during evaluation loops.
             with torch.enable_grad():
                 displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
                 energy = self.compute_energy(displacement_batch)
                 # Compute gradient
                 grad = torch.autograd.grad(
                     outputs=energy.sum(), # Sum needed if energy is batched
                     inputs=displacement_batch,
                     create_graph=torch.is_grad_enabled(), # Preserve graph if needed for higher derivatives
                     retain_graph=True # Keep graph if energy or grad might be used again
                 )[0]
             return grad # Return directly from within the enable_grad block

        # Handle batching within compute_energy
        energy = self.compute_energy(displacement_batch)

        # Compute gradient: dE/du
        # Need to sum energy if it's batched, as grad expects scalar output
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=displacement_batch,
            create_graph=torch.is_grad_enabled(), # Create graph if we need grad of grad (Hessian)
            retain_graph=True # Allows calling backward or grad multiple times if needed elsewhere
        )[0]

        # Return the gradient (often used as -gradient for internal forces)
        return grad


    def compute_div_p(self, displacement_batch):
        """
        Compute the divergence of the first Piola-Kirchhoff (PK1) stress tensor,
        which corresponds to the internal forces (negative gradient of energy).

        Effectively computes -dE/du and reshapes it.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Divergence of P tensor reshaped to [batch_size, num_nodes, 3] or [num_nodes, 3]
        """
         # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
             original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Compute the gradient dE/du using autograd
        # Ensure requires_grad=True for the input to compute_gradient
        if not displacement_batch.requires_grad:
             u_for_grad = displacement_batch.detach().clone().requires_grad_(True)
        else:
             u_for_grad = displacement_batch

        internal_forces_flat = self.compute_gradient(u_for_grad) # Shape: [batch_size, num_nodes*dim]

        # The internal forces are -dE/du. The divergence of PK1 is also -dE/du (in weak form).
        # So, we just need to reshape the negative gradient.
        div_p = internal_forces_flat.view(batch_size, self.num_nodes, self.dim)

        # Return without batch dimension if input wasn't batched
        if original_shape_was_flat:
            div_p = div_p.squeeze(0)

        return div_p


    def compute_PK1(self, displacement_batch):
        """
        Compute the First Piola-Kirchhoff (PK1) stress tensor P at each
        quadrature point for each element.

        P = ∂W/∂F = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            PK1 tensor P [batch_size, num_elements, num_quad_points, 3, 3]
            or [num_elements, num_quad_points, 3, 3] if not batched.
        """
        if not self.precomputed:
             raise RuntimeError("PK1 computation requires precomputed matrices.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
            original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor: [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized PK1 Computation ---
        # Initialize PK1 tensor: [batch_size, num_elements, num_quad_points, 3, 3]
        pk1 = torch.zeros((batch_size, self.num_elements, self.num_quad_points, self.dim, self.dim),
                         dtype=self.dtype, device=self.device)

        # Gather displacements for all elements: [batch_size, num_elements, nodes_per_element, dim]
        # element_indices = self.elements # Shape: [num_elements, nodes_per_element]
        # Use broadcasting and indexing for efficiency
        # Expand elements indices for batch dim: [batch_size, num_elements, nodes_per_element]
        # This step can be memory intensive if num_elements * nodes_per_element is huge.
        # Consider chunking elements if needed.
        element_indices_expanded = self.elements.unsqueeze(0).expand(batch_size, -1, -1) # B x E x N
        # Gather displacements using batch_gather or equivalent logic
        # Need to index u_reshaped [B, num_nodes, D] using indices [B, E, N] -> result [B, E, N, D]
        # This requires gather along node dimension (dim=1)
        # element_disps = torch.gather(u_reshaped.unsqueeze(2).expand(-1, -1, self.nodes_per_element, -1),
        #                              1, element_indices_expanded.unsqueeze(-1).expand(-1, -1, -1, self.dim)) # This seems overly complex
        # Alternative: Replicate u_reshaped and use advanced indexing?
        # Easier approach: Iterate through batch dim, as done in compute_energy

        # Loop over batch samples
        for b in range(batch_size):
             u_sample = u_reshaped[b] # [num_nodes, dim]
             element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

             # Loop over quadrature points
             for q_idx in range(self.num_quad_points):
                 # Get precomputed shape derivatives for this qp: [E, N, D]
                 dN_dx_q = self.dN_dx_all[:, q_idx, :, :]

                 # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
                 grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                 # Compute deformation gradient F = I + grad(u) : [E, D, D]
                 I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                 F = I + grad_u

                 # --- Calculate PK1 for all elements at this qp ---
                 # Compute J = det(F) : [E]
                 J = torch.linalg.det(F)

                 # Clamp J for log stability
                 J_safe = torch.clamp(J, min=self.eps)
                 log_J = torch.log(J_safe)

                 # Compute F inverse transpose: F⁻ᵀ : [E, D, D]
                 try:
                     # Compute inverse first, then transpose for potentially better stability
                     inv_F = torch.linalg.inv(F)
                     inv_F_T = inv_F.transpose(-1, -2)
                 except torch.linalg.LinAlgError:
                      # Handle singular F (e.g., due to element inversion)
                      # print(f"Warning: Singular F encountered in PK1 calculation (batch {b}, qp {q_idx}). Using pseudo-inverse.")
                      # Fallback to pseudo-inverse (slower)
                      inv_F = torch.linalg.pinv(F)
                      inv_F_T = inv_F.transpose(-1, -2)
                      # Set log_J based on pseudo-determinant or handle otherwise?
                      # For simplicity, maybe set PK1 contribution to zero here if J is near zero?
                      zero_pk1_mask = torch.abs(J) < self.eps
                      log_J = torch.log(torch.clamp(J, min=self.eps)) # Still need logJ for non-singular ones

                 # Compute PK1: P = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ
                 # Need to reshape log_J for broadcasting: [E, 1, 1]
                 log_J_reshaped = log_J.unsqueeze(-1).unsqueeze(-1)

                 P_q = self.mu * (F - inv_F_T) + self.lmbda * log_J_reshaped * inv_F_T

                 # Handle potentially singular elements where we used pseudo-inverse
                 # If mask exists, apply it (optional, depends on desired behavior)
                 # if 'zero_pk1_mask' in locals() and torch.any(zero_pk1_mask):
                 #     P_q[zero_pk1_mask] = 0.0

                 # Store result for this batch sample and quad point
                 pk1[b, :, q_idx, :, :] = P_q


        # Return result, removing batch dim if input was flat
        if original_shape_was_flat:
            pk1 = pk1.squeeze(0)

        return pk1


    # --- Volume Comparison (assuming it uses only coordinates and detJ_all) ---
    # Keep the volume comparison methods as they were, assuming they rely on
    # precomputed detJ or calculate volume geometrically.
    # Note: _compute_deformed_volume might need adjustment if it relies on F
    # or if the geometric calculation needs refinement for hex elements.

    def compute_volume_comparison(self, u_linear, u_total):
        """
        Compare volumes between original mesh, linear modes prediction, and neural network prediction.

        Args:
            u_linear: Linear displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]
            u_total: Total displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]

        Returns:
            Dictionary with volume information (volumes and ratios)
        """
        # Handle batch dimension if present (assuming batch=1 for comparison)
        if u_linear.dim() > 1 and u_linear.shape[0] == 1:
            u_linear = u_linear.squeeze(0)
        if u_total.dim() > 1 and u_total.shape[0] == 1:
            u_total = u_total.squeeze(0)

        # Ensure displacements are flat [N*D] for _compute_deformed_volume if needed
        # (Current implementation reshapes inside)

        # Calculate volumes
        original_volume = self._compute_mesh_volume()
        # Note: _compute_deformed_volume currently only works well for Tets.
        # Needs update for Hex using quadrature and J from deformed state.
        linear_volume = self._compute_deformed_volume(u_linear)
        neural_volume = self._compute_deformed_volume(u_total)

        # Calculate volume ratios
        # Add epsilon to prevent division by zero if original volume is tiny
        vol_eps = 1e-12
        linear_volume_ratio = linear_volume / (original_volume + vol_eps)
        neural_volume_ratio = neural_volume / (original_volume + vol_eps)

        # Calculate improvement ratio (how much closer neural is to 1 than linear)
        # Avoid division by zero if linear is already perfect
        linear_deviation = abs(linear_volume_ratio - 1.0)
        neural_deviation = abs(neural_volume_ratio - 1.0)
        if linear_deviation > self.eps: # Use class epsilon
            improvement_ratio = neural_deviation / linear_deviation
        elif neural_deviation <= self.eps : # Both are very close to 1
             improvement_ratio = 0.0 # Perfect preservation or improvement
        else: # Linear was perfect, neural is not
             improvement_ratio = float('inf') # Indicates worsening

        # Create result dictionary
        volume_info = {
            'original_volume': original_volume.item() if isinstance(original_volume, torch.Tensor) else original_volume,
            'linear_volume': linear_volume.item() if isinstance(linear_volume, torch.Tensor) else linear_volume,
            'neural_volume': neural_volume.item() if isinstance(neural_volume, torch.Tensor) else neural_volume,
            'linear_volume_ratio': linear_volume_ratio.item() if isinstance(linear_volume_ratio, torch.Tensor) else linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio.item() if isinstance(neural_volume_ratio, torch.Tensor) else neural_volume_ratio,
            'volume_preservation_improvement_ratio': improvement_ratio # Lower is better (0=perfect neural, 1=same as linear)
        }

        return volume_info

    def _compute_mesh_volume(self):
        """Calculate the total volume of the original undeformed mesh using precomputed detJ."""
        if not self.precomputed or self.detJ_all is None:
             print("Warning: Precomputed detJ not available for volume calculation. Falling back to geometric (Tet only).")
             # Fallback geometric calculation (Tet only)
             if self.nodes_per_element != 4:
                 return torch.tensor(float('nan'), device=self.device) # Indicate failure
             total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)
             for element_nodes in self.elements:
                 vertices = self.coordinates[element_nodes[:4]] # Use first 4 nodes for linear tet volume
                 # Using formula V = |det(v1-v0, v2-v0, v3-v0)| / 6
                 mat = (vertices[1:] - vertices[0]).T # Shape [3, 3]
                 volume = torch.abs(torch.linalg.det(mat)) / 6.0
                 total_volume += volume
             return total_volume

        # Preferred method: Integrate 1 over the domain using quadrature
        # Volume = ∑_elements ∑_qp (1 * detJ * quad_weight)
        # Reshape quad_weights for broadcasting: [1, num_qp]
        quad_weights_r = self.quadrature_weights.unsqueeze(0)
        # Element volumes: sum over quad points: [num_elements]
        element_volumes = torch.sum(self.detJ_all * quad_weights_r, dim=1)
        # Total volume: sum over elements
        total_volume = torch.sum(element_volumes)

        # Ensure positive volume (detJ should be positive for valid meshes)
        return torch.clamp(total_volume, min=0.0)


    def _compute_deformed_volume(self, displacement):
        """
        Calculate the total volume of the mesh after applying a displacement field.
        Uses integration of the determinant of the deformation gradient (J).

        Args:
            displacement: Displacement tensor [num_nodes*dim] or [num_nodes, dim]

        Returns:
            Total deformed volume (scalar tensor).
        """
        if not self.precomputed:
             raise RuntimeError("Deformed volume computation requires precomputed matrices.")

        # Ensure displacement is in the correct shape [num_nodes, dim]
        if displacement.dim() == 1:
             u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes:
             u_sample = displacement
        else:
             raise ValueError(f"Invalid displacement shape: {displacement.shape}")

        # Get element displacements [num_elements, nodes_per_element, dim]
        element_disps = u_sample[self.elements]

        # Initialize volume
        total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Integrate J over the reference domain: ∫_Ω₀ J dV = Volume(Ω)
        # Loop over quadrature points
        for q_idx in range(self.num_quad_points):
            # Get precomputed data for this quad point:
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [E, N, D]
            detJ_ref_q = self.detJ_all[:, q_idx]     # [E] (Jacobian of reference map)
            qw_q = self.quadrature_weights[q_idx]    # scalar

            # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

            # Compute deformation gradient F = I + grad(u) : [E, D, D]
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u

            # Compute J = det(F) for the deformed state : [E]
            J_deformed = torch.linalg.det(F)

            # Add contribution to total volume: ∫ J dV₀ = ∑ J * detJ_ref * quad_weight
            # We integrate J over the *reference* volume elements
            total_volume += torch.sum(J_deformed * detJ_ref_q * qw_q)

        return total_volume




    
class SOFANeoHookeanModel(torch.nn.Module):
    """
    PyTorch-based Neo-Hookean energy model using precomputed mesh data.

    Implements the Neo-Hookean formulation:
    W = 0.5 * μ * (I_C - 3 - 2 * ln(J)) + 0.25 * λ * (J² - 1 - 2 * ln(J))
    """

    # Modify the __init__ method
    def __init__(self, coordinates_np, elements_np, degree, E, nu, precompute_matrices=True, device=None, dtype=torch.float64):
        """
        Initialize with mesh data from NumPy arrays.

        Args:
            coordinates_np: NumPy array of node coordinates [num_nodes, dim]
            elements_np: NumPy array of element connectivity [num_elements, nodes_per_element]
            degree: FEM degree (used for quadrature rules)
            E: Young's modulus
            nu: Poisson's ratio
            precompute_matrices: Whether to precompute FEM matrices (highly recommended)
            device: Computation device (e.g., 'cpu', 'cuda:0')
            dtype: Data type for computation (e.g., torch.float32, torch.float64)
        """
        super(SOFANeoHookeanModel, self).__init__()

        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"TorchNeoHookeanFEM: Using device: {self.device}, dtype: {self.dtype}")

        # Material properties (same as before)
        self.E = torch.tensor(E, dtype=self.dtype, device=self.device)
        self.nu = torch.tensor(nu, dtype=self.dtype, device=self.device)
        # ... (lambda, mu calculation remains the same) ...
        if torch.abs(1.0 - 2.0 * self.nu) < 1e-9:
             print("Warning: nu is close to 0.5. Using a large value for lambda.")
             safe_nu = torch.min(self.nu, torch.tensor(0.49999, dtype=self.dtype, device=self.device))
             self.lmbda = self.E * safe_nu / ((1 + safe_nu) * (1 - 2 * safe_nu))
        else:
             self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.mu = self.E / (2 * (1 + self.nu))
        print(f"Material Properties: E={self.E.item():.2f}, nu={self.nu.item():.2f}, mu={self.mu.item():.2f}, lambda={self.lmbda.item():.2f}")


        # Store degree
        self.degree = degree

        # --- Load mesh data from arrays ---
        self.coordinates = torch.tensor(coordinates_np, device=self.device, dtype=self.dtype)
        self.elements = torch.tensor(elements_np, device=self.device, dtype=torch.long)

        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        self.num_elements = self.elements.shape[0]
        self.nodes_per_element = self.elements.shape[1] # Infer from loaded data

        if self.dim != 3:
            print(f"Warning: Expected 3D coordinates, but got {self.dim}D.")

        print(f"Mesh info loaded: {self.num_nodes} nodes, {self.num_elements} elements")
        print(f"Nodes per element: {self.nodes_per_element} (degree={self.degree})")
        # --- End mesh data loading ---

        # Create element data structure (calls _generate_quadrature and _precompute_derivatives)
        self._setup_elements(precompute_matrices)

        # Save configuration parameters
        self.precompute_matrices = precompute_matrices

        # Small value for safe log/sqrt
        self.eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)


    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        self._generate_quadrature() # Depends on self.nodes_per_element

        if precompute:
            print("Precomputing element shape function derivatives...")
            self._precompute_derivatives()
            self.precomputed = True
            print("Precomputation finished.")
        else:
            print("Derivatives will be computed on-the-fly (less efficient).")
            self.precomputed = False
            # Initialize placeholders if needed later, or handle errors
            self.dN_dx_all = None
            self.detJ_all = None

    def _generate_quadrature(self):
        """Generate quadrature rules based ONLY on nodes_per_element."""
        if self.nodes_per_element == 4:  # Linear Tetrahedron
            # 4-point rule
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                                  dtype=self.dtype,
                                                  device=self.device) / 6.0
            print(f"Using 4-point quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 8: # Linear Hexahedron
            # 2x2x2 Gaussian quadrature
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
                [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
            print(f"Using 8-point (2x2x2 Gauss) quadrature for {self.nodes_per_element}-node elements.")
        # Add rules for other element types if needed (e.g., 10-node Tet, 27-node Hex)
        elif self.nodes_per_element == 10: # Quadratic Tetrahedron
             # Example: 5-point rule (degree 3 exactness) - Adjust points/weights as needed
             print(f"Warning: Using placeholder 5-point quadrature for {self.nodes_per_element}-node Tet. Verify accuracy.")
             # Replace with actual points/weights for quadratic tet
             a = 0.108103018168070
             b = 0.445948490915965
             self.quadrature_points = torch.tensor([
                 [0.25, 0.25, 0.25],
                 [a, a, a],
                 [b, a, a],
                 [a, b, a],
                 [a, a, b]
             ], dtype=self.dtype, device=self.device)
             w0 = -0.8 / 6.0 # -2/15 * (1/6)
             w1 = 0.325 / 6.0 # 13/240 * (1/6)
             self.quadrature_weights = torch.tensor([w0, w1, w1, w1, w1], dtype=self.dtype, device=self.device) * 4.0 # Adjust weights based on rule source

        else:
            raise NotImplementedError(f"Quadrature rule not implemented for {self.nodes_per_element}-node elements.")

        self.num_quad_points = len(self.quadrature_points)


    # Helper to get cell type string robustly after init
    def _get_cell_type_str(self):
         try:
            import dolfinx
            from dolfinx.fem import FunctionSpace
            V_dummy = FunctionSpace(self.domain, ("CG", self.degree))
            return V_dummy.ufl_element().cell().cellname()
         except:
            # Fallback if dolfinx not available later or error
            if self.nodes_per_element == 4: return "tetrahedron"
            if self.nodes_per_element == 8: return "hexahedron"
            return "unknown"


    def _precompute_derivatives(self):
        """Precompute shape function derivatives w.r.t. physical coords (dN/dx)
           and Jacobian determinants (detJ) for all elements and quadrature points."""
        num_qp = self.num_quad_points

        # Allocate tensors
        # Shape: [num_elements, num_quad_points, nodes_per_element, dim]
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, self.dim),
                                    dtype=self.dtype, device=self.device)
        # Shape: [num_elements, num_quad_points]
        self.detJ_all = torch.zeros((self.num_elements, num_qp),
                                   dtype=self.dtype, device=self.device)

        # Process elements - can be done element-wise or batched if memory allows
        # Simpler element-wise loop for clarity:
        for e_idx in range(self.num_elements):
            element_node_indices = self.elements[e_idx]
            element_coords = self.coordinates[element_node_indices] # Coords of nodes for this element

            for q_idx, qp in enumerate(self.quadrature_points):
                # Compute dN/dξ (derivatives in reference element) and then dN/dx
                dN_dxi = self._shape_function_derivatives_ref(qp) # Shape: [nodes_per_element, dim]
                # Compute Jacobian J = dX/dξ = ∑ (Coords_n * dNn/dξ)
                # J_ij = ∑_n (Coords_ni * dNn/dξ_j)
                J = torch.einsum('ni,nj->ij', element_coords, dN_dxi) # Shape: [dim, dim]

                try:
                    detJ = torch.linalg.det(J)
                    invJ = torch.linalg.inv(J)
                except torch.linalg.LinAlgError as err:
                     print(f"Error computing inv/det(J) for element {e_idx} at QP {q_idx}: {err}")
                     print(f"Jacobian Matrix J:\n{J.cpu().numpy()}")
                     # Handle degenerate element - maybe set derivatives to zero or raise error
                     detJ = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                     # Assign zero derivatives to prevent NaN propagation, but this element won't contribute correctly
                     dN_dx = torch.zeros_like(dN_dxi) # Zeros for dN/dx
                     # Or could try pseudo-inverse? invJ = torch.linalg.pinv(J)

                if detJ <= 0:
                     print(f"Warning: Non-positive Jacobian determinant ({detJ.item():.4e}) for element {e_idx} at QP {q_idx}. Check mesh quality.")
                     # Set to small positive value to avoid issues with weights? Or handle upstream.


                # Compute shape function derivatives w.r.t. physical coordinates:
                # dN/dx = dN/dξ * dξ/dX = dN/dξ * J⁻¹
                # dNn/dx_k = ∑_j (dNn/dξ_j * invJ_jk)
                dN_dx = torch.einsum('nj,jk->nk', dN_dxi, invJ) # Shape: [nodes_per_element, dim]

                # Store results
                self.dN_dx_all[e_idx, q_idx] = dN_dx
                self.detJ_all[e_idx, q_idx] = detJ

            # Optional: Add progress indicator for large meshes
            # if (e_idx + 1) % 1000 == 0:
            #     print(f"Precomputing derivatives: {e_idx + 1}/{self.num_elements} elements processed.")

    def _shape_function_derivatives_ref(self, qp_ref):
        """Compute shape function derivatives w.r.t. reference coordinates (ξ, η, ζ)
           at a given reference quadrature point qp_ref."""
        # qp_ref is a tensor [dim] with coordinates in the reference element

        if self.nodes_per_element == 4: # Linear Tetrahedron (Ref coords: L1, L2, L3, L4 sum to 1)
             # Derivatives are constant for linear tet
             # Using barycentric derivatives: dNi/dξj where ξ = (ξ, η, ζ) mapped from barycentric
             # dN1/dξ = -1, dN1/dη = -1, dN1/dζ = -1
             # dN2/dξ = 1,  dN2/dη = 0,  dN2/dζ = 0
             # dN3/dξ = 0,  dN3/dη = 1,  dN3/dζ = 0
             # dN4/dξ = 0,  dN4/dη = 0,  dN4/dζ = 1
             dN_dxi = torch.tensor([
                 [-1.0, -1.0, -1.0],
                 [ 1.0,  0.0,  0.0],
                 [ 0.0,  1.0,  0.0],
                 [ 0.0,  0.0,  1.0]
             ], dtype=self.dtype, device=self.device)

        elif self.nodes_per_element == 8: # Linear Hexahedron (Ref coords: ξ, η, ζ in [-1, 1])
            xi, eta, zeta = qp_ref[0], qp_ref[1], qp_ref[2]

            # Precompute terms
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            xim = one - xi
            xip = one + xi
            etam = one - eta
            etap = one + eta
            zetam = one - zeta
            zetap = one + zeta

            # Shape function derivatives at this point (dN/dξ, dN/dη, dN/dζ)
            # Shape: [nodes_per_element=8, dim=3]
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            eighth = 0.125

            # Derivatives w.r.t. xi (d/dξ)
            dN_dxi[0, 0] = -eighth * etam * zetam
            dN_dxi[1, 0] =  eighth * etam * zetam
            dN_dxi[2, 0] =  eighth * etap * zetam
            dN_dxi[3, 0] = -eighth * etap * zetam
            dN_dxi[4, 0] = -eighth * etam * zetap
            dN_dxi[5, 0] =  eighth * etam * zetap
            dN_dxi[6, 0] =  eighth * etap * zetap
            dN_dxi[7, 0] = -eighth * etap * zetap

            # Derivatives w.r.t. eta (d/dη)
            dN_dxi[0, 1] = -eighth * xim * zetam
            dN_dxi[1, 1] = -eighth * xip * zetam
            dN_dxi[2, 1] =  eighth * xip * zetam
            dN_dxi[3, 1] =  eighth * xim * zetam
            dN_dxi[4, 1] = -eighth * xim * zetap
            dN_dxi[5, 1] = -eighth * xip * zetap
            dN_dxi[6, 1] =  eighth * xip * zetap
            dN_dxi[7, 1] =  eighth * xim * zetap

            # Derivatives w.r.t. zeta (d/dζ)
            dN_dxi[0, 2] = -eighth * xim * etam
            dN_dxi[1, 2] = -eighth * xip * etam
            dN_dxi[2, 2] = -eighth * xip * etap
            dN_dxi[3, 2] = -eighth * xim * etap
            dN_dxi[4, 2] =  eighth * xim * etam
            dN_dxi[5, 2] =  eighth * xip * etam
            dN_dxi[6, 2] =  eighth * xip * etap
            dN_dxi[7, 2] =  eighth * xim * etap

        else:
            # Needs implementation for other element types (e.g., quadratic tet/hex)
            raise NotImplementedError(f"Shape function derivatives not implemented for {self.nodes_per_element}-node elements.")

        return dN_dxi


    def forward(self, u_tensor):
        """
        PyTorch forward method - computes total energy.

        Args:
            u_tensor: Displacement field [num_nodes*dim] or [batch_size, num_nodes*dim]

        Returns:
            Total strain energy (scalar for single input, tensor [batch_size] for batch)
        """
        return self.compute_energy(u_tensor)

    def compute_energy(self, displacement_batch):
        """
        Compute total elastic energy for displacement field(s).

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Total strain energy per batch sample [batch_size] or scalar
        """
        if not self.precomputed:
             raise RuntimeError("Energy computation requires precomputed matrices. Initialize with precompute_matrices=True.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor to [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized Energy Computation ---
        # Gather displacements for all elements: [batch_size, num_elements, nodes_per_element, dim]
        # Need efficient gathering. self.elements gives indices [num_elements, nodes_per_element]
        # u_reshaped is [batch_size, num_nodes, dim]
        # We can use advanced indexing, but need to handle batch dimension.
        # Option 1: Loop over batch (simple, might be okay if batch_size is small)
        # Option 2: Expand elements indices and gather (more complex, potentially faster)

        # Option 1: Loop over batch (simpler to implement/debug)
        total_energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        for b in range(batch_size):
            u_sample = u_reshaped[b] # [num_nodes, dim]
            element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

            # Initialize energy for this sample
            energy_sample = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            # Loop over quadrature points
            for q_idx in range(self.num_quad_points):
                # Get precomputed data for this quad point:
                dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [num_elements, nodes_per_element, dim]
                detJ_q = self.detJ_all[:, q_idx]         # [num_elements]
                qw_q = self.quadrature_weights[q_idx]    # scalar

                # Compute gradient of displacement for all elements at this qp:
                # grad(u)_ij = ∑_n (u_nj * dNn/dx_i)
                # Input element_disps: [E, N, D]
                # Input dN_dx_q:      [E, N, D]
                # Output grad_u:      [E, D, D]
                grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                # Compute deformation gradient F = I + grad(u)
                # Need identity matrix shaped [E, D, D]
                I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                F = I + grad_u # [num_elements, dim, dim]

                # Compute energy density at this quadrature point for all elements
                # Need F with shape [*, 3, 3] for the density function
                energy_density_q = self._compute_neohookean_energy_density(F) # [num_elements]

                # Add weighted contribution to total energy for this sample
                # dEnergy = W * detJ * quad_weight
                energy_sample += torch.sum(energy_density_q * detJ_q * qw_q)

            total_energy[b] = energy_sample

        # Return single value if input wasn't batched
        if not is_batch:
            total_energy = total_energy.squeeze(0)

        return total_energy


    def _compute_neohookean_energy_density(self, F):
        """
        Compute Neo-Hookean strain energy density (UFL-equivalent formulation).

        Args:
            F: Deformation gradient tensor [*, 3, 3] (batch dimension handled by *)

        Returns:
            Energy density W [*]
        """
        # Ensure F has the correct dimensions (at least 3x3)
        if F.shape[-2:] != (3, 3):
             raise ValueError(f"Input F must have shape [..., 3, 3], but got {F.shape}")

        # Determinant J = det(F)
        J = torch.linalg.det(F)

        # Check for non-positive J (potential element inversion)
        if torch.any(J <= self.eps): # Use small epsilon threshold
             # print(f"Warning: Non-positive Jacobian detected (min J = {torch.min(J).item()}). Clamping J for log.")
             # Clamp J to a small positive value for stability in log
             J_safe = torch.clamp(J, min=self.eps)
        else:
             J_safe = J

        # Logarithm of J
        log_J = torch.log(J_safe)

        # First invariant of Right Cauchy-Green C = FᵀF
        # I_C = tr(C) = ∑_i,j (F_ji * F_ji)
        IC = torch.einsum('...ji,...ji->...', F, F) # Sum over last two dimensions

        # Neo-Hookean energy density (W₂)
        # W = (μ/2) * (I_C - 3 - 2*ln(J)) + (λ/4) * (J² - 1 - 2*ln(J))
        W = 0.5 * self.mu * (IC - 3.0 - 2.0 * log_J) + 0.25 * self.lmbda * (J ** 2 - 1.0 - 2.0 * log_J)
        return W

    # --- Methods for Gradients and Stress ---

    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy w.r.t. displacements).

        Uses torch.autograd for automatic differentiation.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Internal forces (nodal forces) [-dE/du] with the same shape as input.
        """
        # Ensure input requires grad
        if not displacement_batch.requires_grad:
            displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
        elif not torch.is_grad_enabled():
             # If called within torch.no_grad(), autograd won't work.
             # Re-enable grad temporarily for this computation.
             # This might be needed if this function is called during evaluation loops.
             with torch.enable_grad():
                 displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
                 energy = self.compute_energy(displacement_batch)
                 # Compute gradient
                 grad = torch.autograd.grad(
                     outputs=energy.sum(), # Sum needed if energy is batched
                     inputs=displacement_batch,
                     create_graph=torch.is_grad_enabled(), # Preserve graph if needed for higher derivatives
                     retain_graph=True # Keep graph if energy or grad might be used again
                 )[0]
             return grad # Return directly from within the enable_grad block

        # Handle batching within compute_energy
        energy = self.compute_energy(displacement_batch)

        # Compute gradient: dE/du
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=displacement_batch,
            create_graph=torch.is_grad_enabled(), # Create graph if we need grad of grad (Hessian)
            retain_graph=True # Allows calling backward or grad multiple times if needed elsewhere
        )[0]

        # Return the gradient (often used as -gradient for internal forces)
        return grad


    def compute_div_p(self, displacement_batch):
        """
        Compute the divergence of the first Piola-Kirchhoff (PK1) stress tensor,
        which corresponds to the internal forces (negative gradient of energy).

        Effectively computes -dE/du and reshapes it.

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            Divergence of P tensor reshaped to [batch_size, num_nodes, 3] or [num_nodes, 3]
        """
         # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
             original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Compute the gradient dE/du using autograd
        # Ensure requires_grad=True for the input to compute_gradient
        if not displacement_batch.requires_grad:
             u_for_grad = displacement_batch.detach().clone().requires_grad_(True)
        else:
             u_for_grad = displacement_batch

        internal_forces_flat = self.compute_gradient(u_for_grad) # Shape: [batch_size, num_nodes*dim]

        # The internal forces are -dE/du. The divergence of PK1 is also -dE/du (in weak form).
        div_p = internal_forces_flat.view(batch_size, self.num_nodes, self.dim)

        # Return without batch dimension if input wasn't batched
        if original_shape_was_flat:
            div_p = div_p.squeeze(0)

        return div_p


    def compute_PK1(self, displacement_batch):
        """
        Compute the First Piola-Kirchhoff (PK1) stress tensor P at each
        quadrature point for each element.

        P = ∂W/∂F = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ

        Args:
            displacement_batch: Displacement tensor [batch_size, num_nodes*dim] or [num_nodes*dim]

        Returns:
            PK1 tensor P [batch_size, num_elements, num_quad_points, 3, 3]
            or [num_elements, num_quad_points, 3, 3] if not batched.
        """
        if not self.precomputed:
             raise RuntimeError("PK1 computation requires precomputed matrices.")

        # Handle input dimensionality
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
            original_shape_was_flat = False

        batch_size = displacement_batch.shape[0]

        # Reshape displacement tensor: [batch_size, num_nodes, dim]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        # --- Vectorized PK1 Computation ---
        # Initialize PK1 tensor: [batch_size, num_elements, num_quad_points, 3, 3]
        pk1 = torch.zeros((batch_size, self.num_elements, self.num_quad_points, self.dim, self.dim),
                         dtype=self.dtype, device=self.device)


        # Loop over batch samples
        for b in range(batch_size):
             u_sample = u_reshaped[b] # [num_nodes, dim]
             element_disps = u_sample[self.elements] # [num_elements, nodes_per_element, dim]

             # Loop over quadrature points
             for q_idx in range(self.num_quad_points):
                 # Get precomputed shape derivatives for this qp: [E, N, D]
                 dN_dx_q = self.dN_dx_all[:, q_idx, :, :]

                 # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
                 grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

                 # Compute deformation gradient F = I + grad(u) : [E, D, D]
                 I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                 F = I + grad_u

                 # --- Calculate PK1 for all elements at this qp ---
                 # Compute J = det(F) : [E]
                 J = torch.linalg.det(F)

                 # Clamp J for log stability
                 J_safe = torch.clamp(J, min=self.eps)
                 log_J = torch.log(J_safe)

                 # Compute F inverse transpose: F⁻ᵀ : [E, D, D]
                 try:
                     # Compute inverse first, then transpose for potentially better stability
                     inv_F = torch.linalg.inv(F)
                     inv_F_T = inv_F.transpose(-1, -2)
                 except torch.linalg.LinAlgError:
                      # Handle singular F (e.g., due to element inversion)
                      # print(f"Warning: Singular F encountered in PK1 calculation (batch {b}, qp {q_idx}). Using pseudo-inverse.")
                      # Fallback to pseudo-inverse (slower)
                      inv_F = torch.linalg.pinv(F)
                      inv_F_T = inv_F.transpose(-1, -2)
                      # Set log_J based on pseudo-determinant or handle otherwise?
                      # For simplicity, maybe set PK1 contribution to zero here if J is near zero?
                      zero_pk1_mask = torch.abs(J) < self.eps
                      log_J = torch.log(torch.clamp(J, min=self.eps)) # Still need logJ for non-singular ones

                 # Compute PK1: P = μ * (F - F⁻ᵀ) + λ * ln(J) * F⁻ᵀ
                 # Need to reshape log_J for broadcasting: [E, 1, 1]
                 log_J_reshaped = log_J.unsqueeze(-1).unsqueeze(-1)

                 P_q = self.mu * (F - inv_F_T) + self.lmbda * log_J_reshaped * inv_F_T

           

                 # Store result for this batch sample and quad point
                 pk1[b, :, q_idx, :, :] = P_q


        if original_shape_was_flat:
            pk1 = pk1.squeeze(0)

        return pk1


    # --- Volume Comparison ---

    def compute_volume_comparison(self, u_linear, u_total):
        """
        Compare volumes between original mesh, linear modes prediction, and neural network prediction.

        Args:
            u_linear: Linear displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]
            u_total: Total displacement field [num_nodes*dim] or [batch=1, num_nodes*dim]

        Returns:
            Dictionary with volume information (volumes and ratios)
        """
        # Handle batch dimension if present (assuming batch=1 for comparison)
        if u_linear.dim() > 1 and u_linear.shape[0] == 1:
            u_linear = u_linear.squeeze(0)
        if u_total.dim() > 1 and u_total.shape[0] == 1:
            u_total = u_total.squeeze(0)

        # Ensure displacements are flat [N*D] for _compute_deformed_volume if needed

        # Calculate volumes
        original_volume = self._compute_mesh_volume()
        linear_volume = self._compute_deformed_volume(u_linear)
        neural_volume = self._compute_deformed_volume(u_total)

        # Calculate volume ratios
        # Add epsilon to prevent division by zero if original volume is tiny
        vol_eps = 1e-12
        linear_volume_ratio = linear_volume / (original_volume + vol_eps)
        neural_volume_ratio = neural_volume / (original_volume + vol_eps)

        # Calculate improvement ratio (how much closer neural is to 1 than linear)
        # Avoid division by zero if linear is already perfect
        linear_deviation = abs(linear_volume_ratio - 1.0)
        neural_deviation = abs(neural_volume_ratio - 1.0)
        if linear_deviation > self.eps: # Use class epsilon
            improvement_ratio = neural_deviation / linear_deviation
        elif neural_deviation <= self.eps : # Both are very close to 1
             improvement_ratio = 0.0 # Perfect preservation or improvement
        else: # Linear was perfect, neural is not
             improvement_ratio = float('inf') # Indicates worsening

        # Create result dictionary
        volume_info = {
            'original_volume': original_volume.item() if isinstance(original_volume, torch.Tensor) else original_volume,
            'linear_volume': linear_volume.item() if isinstance(linear_volume, torch.Tensor) else linear_volume,
            'neural_volume': neural_volume.item() if isinstance(neural_volume, torch.Tensor) else neural_volume,
            'linear_volume_ratio': linear_volume_ratio.item() if isinstance(linear_volume_ratio, torch.Tensor) else linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio.item() if isinstance(neural_volume_ratio, torch.Tensor) else neural_volume_ratio,
            'volume_preservation_improvement_ratio': improvement_ratio # Lower is better (0=perfect neural, 1=same as linear)
        }

        return volume_info

    def _compute_mesh_volume(self):
        """Calculate the total volume of the original undeformed mesh using precomputed detJ."""
        if not self.precomputed or self.detJ_all is None:
             print("Warning: Precomputed detJ not available for volume calculation. Falling back to geometric (Tet only).")
             # Fallback geometric calculation (Tet only)
             if self.nodes_per_element != 4:
                 return torch.tensor(float('nan'), device=self.device) # Indicate failure
             total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)
             for element_nodes in self.elements:
                 vertices = self.coordinates[element_nodes[:4]] # Use first 4 nodes for linear tet volume
                 # Using formula V = |det(v1-v0, v2-v0, v3-v0)| / 6
                 mat = (vertices[1:] - vertices[0]).T # Shape [3, 3]
                 volume = torch.abs(torch.linalg.det(mat)) / 6.0
                 total_volume += volume
             return total_volume

        # Integrate 1 over the domain using quadrature
        # Volume = ∑_elements ∑_qp (1 * detJ * quad_weight)
        # Reshape quad_weights for broadcasting: [1, num_qp]
        quad_weights_r = self.quadrature_weights.unsqueeze(0)
        # Element volumes: sum over quad points: [num_elements]
        element_volumes = torch.sum(self.detJ_all * quad_weights_r, dim=1)
        # Total volume: sum over elements
        total_volume = torch.sum(element_volumes)

        # Ensure positive volume (detJ should be positive for valid meshes)
        return torch.clamp(total_volume, min=0.0)


    def _compute_deformed_volume(self, displacement):
        """
        Calculate the total volume of the mesh after applying a displacement field.
        Uses integration of the determinant of the deformation gradient (J).

        Args:
            displacement: Displacement tensor [num_nodes*dim] or [num_nodes, dim]

        Returns:
            Total deformed volume (scalar tensor).
        """
        if not self.precomputed:
             raise RuntimeError("Deformed volume computation requires precomputed matrices.")

        # Ensure displacement is in the correct shape [num_nodes, dim]
        if displacement.dim() == 1:
             u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes:
             u_sample = displacement
        else:
             raise ValueError(f"Invalid displacement shape: {displacement.shape}")

        # Get element displacements [num_elements, nodes_per_element, dim]
        element_disps = u_sample[self.elements]

        # Initialize volume
        total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)

        # Integrate J over the reference domain: ∫_Ω₀ J dV = Volume(Ω)
        # Loop over quadrature points
        for q_idx in range(self.num_quad_points):
            # Get precomputed data for this quad point:
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :] # [E, N, D]
            detJ_ref_q = self.detJ_all[:, q_idx]     # [E] (Jacobian of reference map)
            qw_q = self.quadrature_weights[q_idx]    # scalar

            # Compute grad(u) = ∑ u_n * dNn/dx : [E, D, D]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

            # Compute deformation gradient F = I + grad(u) : [E, D, D]
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u

            # Compute J = det(F) for the deformed state : [E]
            J_deformed = torch.linalg.det(F)

            # Add contribution to total volume: ∫ J dV₀ = ∑ J * detJ_ref * quad_weight
            # We integrate J over the *reference* volume elements
            total_volume += torch.sum(J_deformed * detJ_ref_q * qw_q)

        return total_volume
    
    def compute_deformation_gradients(self, displacement):
        """
        Compute the deformation gradients (F) for a given displacement field.

        Args:
            displacement: Displacement tensor [num_nodes*dim] or [num_nodes, dim]

        Returns:
            Deformation gradients F [num_elements, num_quad_points, dim, dim]
        """
        if not self.precomputed:
            raise RuntimeError("Deformation gradient computation requires precomputed matrices.")

        # Ensure displacement is in the correct shape [num_nodes, dim]
        if displacement.dim() == 1:
            u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes:
            u_sample = displacement
        else:
            raise ValueError(f"Invalid displacement shape: {displacement.shape}")

        # Get element displacements [num_elements, nodes_per_element, dim]
        element_disps = u_sample[self.elements]

        # Initialize deformation gradients tensor
        deformation_gradients = torch.zeros(
            (self.num_elements, self.num_quad_points, self.dim, self.dim),
            dtype=self.dtype, device=self.device
        )

        # Loop over quadrature points
        for q_idx in range(self.num_quad_points):
            # Get precomputed data for this quad point:
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :]  # [num_elements, nodes_per_element, dim]

            # Compute grad(u) = ∑ u_n * dNn/dx : [num_elements, dim, dim]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

            # Compute deformation gradient F = I + grad(u) : [num_elements, dim, dim]
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u

            # Store deformation gradients for this quadrature point
            deformation_gradients[:, q_idx, :, :] = F

        return deformation_gradients




class SOFAStVenantKirchhoffModel(torch.nn.Module):
    """
    PyTorch-based St. Venant-Kirchhoff (StVK) energy model using precomputed mesh data.

    Implements the StVK formulation:
    W = 0.5 * λ * (tr(E))² + μ * tr(E²)
    where E = 0.5 * (FᵀF - I) is the Green-Lagrange strain tensor.
    """

    def __init__(self, coordinates_np, elements_np, degree, E, nu, precompute_matrices=True, device=None, dtype=torch.float64):
        """
        Initialize with mesh data from NumPy arrays.

        Args:
            coordinates_np: NumPy array of node coordinates [num_nodes, dim]
            elements_np: NumPy array of element connectivity [num_elements, nodes_per_element]
            degree: FEM degree (used for quadrature rules)
            E: Young's modulus
            nu: Poisson's ratio
            precompute_matrices: Whether to precompute FEM matrices (highly recommended)
            device: Computation device (e.g., 'cpu', 'cuda:0')
            dtype: Data type for computation (e.g., torch.float32, torch.float64)
        """
        super(SOFAStVenantKirchhoffModel, self).__init__()

        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"SOFAStVenantKirchhoffModel: Using device: {self.device}, dtype: {self.dtype}")

        # Material properties (Lamé parameters)
        self.E = torch.tensor(E, dtype=self.dtype, device=self.device)
        self.nu = torch.tensor(nu, dtype=self.dtype, device=self.device)
        if torch.abs(1.0 - 2.0 * self.nu) < 1e-9 or torch.abs(1.0 + self.nu) < 1e-9:
             print("Warning: nu is close to 0.5 or -1. Adjusting Lamé parameters calculation.")
             # Use safe nu to avoid division by zero
             safe_nu = torch.clamp(self.nu, min=-0.99999, max=0.49999)
             self.lmbda = self.E * safe_nu / ((1 + safe_nu) * (1 - 2 * safe_nu))
             self.mu = self.E / (2 * (1 + safe_nu))
        else:
             self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) # First Lamé parameter
             self.mu = self.E / (2 * (1 + self.nu))  # Shear modulus (Second Lamé parameter)
        print(f"Material Properties: E={self.E.item():.2f}, nu={self.nu.item():.2f}, mu={self.mu.item():.2f}, lambda={self.lmbda.item():.2f}")

        # Store degree
        self.degree = degree

        # --- Load mesh data from arrays ---
        self.coordinates = torch.tensor(coordinates_np, device=self.device, dtype=self.dtype)
        self.elements = torch.tensor(elements_np, device=self.device, dtype=torch.long)

        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        self.num_elements = self.elements.shape[0]
        self.nodes_per_element = self.elements.shape[1] # Infer from loaded data

        if self.dim != 3:
            print(f"Warning: Expected 3D coordinates, but got {self.dim}D.")

        print(f"Mesh info loaded: {self.num_nodes} nodes, {self.num_elements} elements")
        print(f"Nodes per element: {self.nodes_per_element} (degree={self.degree})")
        # --- End mesh data loading ---

        # Create element data structure (calls _generate_quadrature and _precompute_derivatives)
        self._setup_elements(precompute_matrices)

        # Save configuration parameters
        self.precompute_matrices = precompute_matrices

        # Small value for stability if needed (less critical for StVK than log(J))
        self.eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)


    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        self._generate_quadrature() # Depends on self.nodes_per_element

        if precompute:
            print("Precomputing element shape function derivatives...")
            self._precompute_derivatives()
            self.precomputed = True
            print("Precomputation finished.")
        else:
            print("Derivatives will be computed on-the-fly (less efficient).")
            self.precomputed = False
            # Initialize placeholders if needed later, or handle errors
            self.dN_dx_all = None
            self.detJ_all = None

    def _generate_quadrature(self):
        """Generate quadrature rules based ONLY on nodes_per_element."""
        if self.nodes_per_element == 4:  # Linear Tetrahedron
            # 4-point rule
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                                  dtype=self.dtype,
                                                  device=self.device) / 6.0
            print(f"Using 4-point quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 8: # Linear Hexahedron
            # 2x2x2 Gaussian quadrature
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
                [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
            print(f"Using 8-point (2x2x2 Gauss) quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 10: # Quadratic Tetrahedron
             print(f"Warning: Using placeholder 5-point quadrature for {self.nodes_per_element}-node Tet. Verify accuracy.")
             a = 0.108103018168070
             b = 0.445948490915965
             self.quadrature_points = torch.tensor([
                 [0.25, 0.25, 0.25], [a, a, a], [b, a, a], [a, b, a], [a, a, b]
             ], dtype=self.dtype, device=self.device)
             w0 = -0.8 / 6.0; w1 = 0.325 / 6.0
             self.quadrature_weights = torch.tensor([w0, w1, w1, w1, w1], dtype=self.dtype, device=self.device) * 4.0
        else:
            raise NotImplementedError(f"Quadrature rule not implemented for {self.nodes_per_element}-node elements.")
        self.num_quad_points = len(self.quadrature_points)


    def _precompute_derivatives(self):
        """Precompute shape function derivatives w.r.t. physical coords (dN/dx)
           and Jacobian determinants (detJ) for all elements and quadrature points."""
        num_qp = self.num_quad_points
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, self.dim),
                                    dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp),
                                   dtype=self.dtype, device=self.device)
        for e_idx in range(self.num_elements):
            element_node_indices = self.elements[e_idx]
            element_coords = self.coordinates[element_node_indices]
            for q_idx, qp in enumerate(self.quadrature_points):
                dN_dxi = self._shape_function_derivatives_ref(qp)
                J = torch.einsum('ni,nj->ij', element_coords, dN_dxi)
                try:
                    detJ = torch.linalg.det(J)
                    invJ = torch.linalg.inv(J)
                except torch.linalg.LinAlgError as err:
                     print(f"Error computing inv/det(J) for element {e_idx} at QP {q_idx}: {err}")
                     detJ = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                     invJ = torch.zeros_like(J) # Assign zero inverse to avoid NaN
                if detJ <= 0:
                     print(f"Warning: Non-positive Jacobian determinant ({detJ.item():.4e}) for element {e_idx} at QP {q_idx}. Check mesh quality.")
                dN_dx = torch.einsum('nj,jk->nk', dN_dxi, invJ)
                self.dN_dx_all[e_idx, q_idx] = dN_dx
                self.detJ_all[e_idx, q_idx] = detJ


    def _shape_function_derivatives_ref(self, qp_ref):
        """Compute shape function derivatives w.r.t. reference coordinates (ξ, η, ζ)
           at a given reference quadrature point qp_ref."""
        # --- This function is identical to the one in SOFANeoHookeanModel ---
        if self.nodes_per_element == 4: # Linear Tetrahedron
             dN_dxi = torch.tensor([
                 [-1.0, -1.0, -1.0], [ 1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  0.0,  1.0]
             ], dtype=self.dtype, device=self.device)
        elif self.nodes_per_element == 8: # Linear Hexahedron
            xi, eta, zeta = qp_ref[0], qp_ref[1], qp_ref[2]
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            xim, xip, etam, etap, zetam, zetap = one-xi, one+xi, one-eta, one+eta, one-zeta, one+zeta
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            eighth = 0.125
            dN_dxi[0, 0] = -eighth * etam * zetam; dN_dxi[1, 0] =  eighth * etam * zetam; dN_dxi[2, 0] =  eighth * etap * zetam; dN_dxi[3, 0] = -eighth * etap * zetam
            dN_dxi[4, 0] = -eighth * etam * zetap; dN_dxi[5, 0] =  eighth * etam * zetap; dN_dxi[6, 0] =  eighth * etap * zetap; dN_dxi[7, 0] = -eighth * etap * zetap
            dN_dxi[0, 1] = -eighth * xim * zetam; dN_dxi[1, 1] = -eighth * xip * zetam; dN_dxi[2, 1] =  eighth * xip * zetam; dN_dxi[3, 1] =  eighth * xim * zetam
            dN_dxi[4, 1] = -eighth * xim * zetap; dN_dxi[5, 1] = -eighth * xip * zetap; dN_dxi[6, 1] =  eighth * xip * zetap; dN_dxi[7, 1] =  eighth * xim * zetap
            dN_dxi[0, 2] = -eighth * xim * etam; dN_dxi[1, 2] = -eighth * xip * etam; dN_dxi[2, 2] = -eighth * xip * etap; dN_dxi[3, 2] = -eighth * xim * etap
            dN_dxi[4, 2] =  eighth * xim * etam; dN_dxi[5, 2] =  eighth * xip * etam; dN_dxi[6, 2] =  eighth * xip * etap; dN_dxi[7, 2] =  eighth * xim * etap
        else:
            raise NotImplementedError(f"Shape function derivatives not implemented for {self.nodes_per_element}-node elements.")
        return dN_dxi


    def forward(self, u_tensor):
        """PyTorch forward method - computes total energy."""
        return self.compute_energy(u_tensor)

    def compute_energy(self, displacement_batch):
        """Compute total elastic energy for displacement field(s)."""
        if not self.precomputed:
             raise RuntimeError("Energy computation requires precomputed matrices.")

        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
        batch_size = displacement_batch.shape[0]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        total_energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        for b in range(batch_size):
            u_sample = u_reshaped[b]
            element_disps = u_sample[self.elements]
            energy_sample = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            for q_idx in range(self.num_quad_points):
                dN_dx_q = self.dN_dx_all[:, q_idx, :, :]
                detJ_q = self.detJ_all[:, q_idx]
                qw_q = self.quadrature_weights[q_idx]

                grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)
                I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                F = I + grad_u

                # Compute StVK energy density
                energy_density_q = self._compute_stvk_energy_density(F) # [num_elements]

                # Add weighted contribution: dEnergy = W * detJ_ref * quad_weight
                energy_sample += torch.sum(energy_density_q * detJ_q * qw_q)

            total_energy[b] = energy_sample

        if not is_batch:
            total_energy = total_energy.squeeze(0)
        return total_energy


    def _compute_stvk_energy_density(self, F):
        """
        Compute St. Venant-Kirchhoff (StVK) strain energy density.

        Args:
            F: Deformation gradient tensor [*, 3, 3]

        Returns:
            Energy density W [*]
        """
        if F.shape[-2:] != (3, 3):
             raise ValueError(f"Input F must have shape [..., 3, 3], but got {F.shape}")

        batch_dims = F.shape[:-2]
        I = torch.eye(3, dtype=self.dtype, device=self.device).expand(*batch_dims, -1, -1)

        # Right Cauchy-Green tensor C = FᵀF [*, 3, 3]
        C = torch.einsum('...ji,...jk->...ik', F, F) # F.transpose(-1, -2) @ F

        # Green-Lagrange strain tensor E = 0.5 * (C - I) [*, 3, 3]
        E = 0.5 * (C - I)

        # Trace of E: tr(E) [*]
        # Use torch.diagonal and sum for batched trace
        trE = torch.sum(torch.diagonal(E, dim1=-2, dim2=-1), dim=-1)

        # Trace of E squared: tr(E²) = ∑_i,j (E_ij * E_ij) [*]
        trE2 = torch.einsum('...ij,...ij->...', E, E)

        # StVK energy density W = 0.5 * λ * (tr(E))² + μ * tr(E²) [*]
        W = 0.5 * self.lmbda * (trE ** 2) + self.mu * trE2
        return W

    # --- Methods for Gradients and Stress ---

    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy w.r.t. displacements).
        Uses torch.autograd for automatic differentiation.
        """
        if not displacement_batch.requires_grad:
            displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
        elif not torch.is_grad_enabled():
             with torch.enable_grad():
                 displacement_batch = displacement_batch.detach().clone().requires_grad_(True)
                 energy = self.compute_energy(displacement_batch)
                 grad = torch.autograd.grad(
                     outputs=energy.sum(), inputs=displacement_batch,
                     create_graph=torch.is_grad_enabled(), retain_graph=True
                 )[0]
             return grad
        energy = self.compute_energy(displacement_batch)
        grad = torch.autograd.grad(
            outputs=energy.sum(), inputs=displacement_batch,
            create_graph=torch.is_grad_enabled(), retain_graph=True
        )[0]
        return grad


    def compute_div_p(self, displacement_batch):
        """
        Compute the divergence of the first Piola-Kirchhoff (PK1) stress tensor,
        which corresponds to the internal forces (-dE/du).
        """
        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
             original_shape_was_flat = False
        batch_size = displacement_batch.shape[0]
        if not displacement_batch.requires_grad:
             u_for_grad = displacement_batch.detach().clone().requires_grad_(True)
        else:
             u_for_grad = displacement_batch
        internal_forces_flat = self.compute_gradient(u_for_grad)
        div_p = internal_forces_flat.view(batch_size, self.num_nodes, self.dim)
        if original_shape_was_flat:
            div_p = div_p.squeeze(0)
        return div_p


    def compute_PK1(self, displacement_batch):
        """
        Compute the First Piola-Kirchhoff (PK1) stress tensor P for StVK.
        P = F @ S, where S = λ * tr(E) * I + 2 * μ * E
        """
        if not self.precomputed:
             raise RuntimeError("PK1 computation requires precomputed matrices.")

        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            original_shape_was_flat = True
            displacement_batch = displacement_batch.unsqueeze(0)
        else:
            original_shape_was_flat = False
        batch_size = displacement_batch.shape[0]
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)

        pk1 = torch.zeros((batch_size, self.num_elements, self.num_quad_points, self.dim, self.dim),
                         dtype=self.dtype, device=self.device)

        for b in range(batch_size):
             u_sample = u_reshaped[b]
             element_disps = u_sample[self.elements]
             for q_idx in range(self.num_quad_points):
                 dN_dx_q = self.dN_dx_all[:, q_idx, :, :]
                 grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)
                 I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                 F = I + grad_u

                 # --- Calculate PK1 for StVK ---
                 C = torch.einsum('...ji,...jk->...ik', F, F)
                 E = 0.5 * (C - I)
                 trE = torch.sum(torch.diagonal(E, dim1=-2, dim2=-1), dim=-1)
                 trE_reshaped = trE.unsqueeze(-1).unsqueeze(-1)
                 S = self.lmbda * trE_reshaped * I + 2.0 * self.mu * E
                 P_q = torch.einsum('...ij,...jk->...ik', F, S) # F @ S
                 # --- End PK1 Calculation ---

                 pk1[b, :, q_idx, :, :] = P_q

        if original_shape_was_flat:
            pk1 = pk1.squeeze(0)
        return pk1

    # --- Volume Comparison Methods ---

    def compute_volume_comparison(self, u_linear, u_total):
        """Compare volumes between original, linear, and total displacements."""
        if u_linear.dim() > 1 and u_linear.shape[0] == 1: u_linear = u_linear.squeeze(0)
        if u_total.dim() > 1 and u_total.shape[0] == 1: u_total = u_total.squeeze(0)
        original_volume = self._compute_mesh_volume()
        linear_volume = self._compute_deformed_volume(u_linear)
        neural_volume = self._compute_deformed_volume(u_total)
        vol_eps = 1e-12
        linear_volume_ratio = linear_volume / (original_volume + vol_eps)
        neural_volume_ratio = neural_volume / (original_volume + vol_eps)
        linear_deviation = abs(linear_volume_ratio - 1.0)
        neural_deviation = abs(neural_volume_ratio - 1.0)
        if linear_deviation > self.eps: improvement_ratio = neural_deviation / linear_deviation
        elif neural_deviation <= self.eps : improvement_ratio = 0.0
        else: improvement_ratio = float('inf')
        volume_info = {
            'original_volume': original_volume.item() if isinstance(original_volume, torch.Tensor) else original_volume,
            'linear_volume': linear_volume.item() if isinstance(linear_volume, torch.Tensor) else linear_volume,
            'neural_volume': neural_volume.item() if isinstance(neural_volume, torch.Tensor) else neural_volume,
            'linear_volume_ratio': linear_volume_ratio.item() if isinstance(linear_volume_ratio, torch.Tensor) else linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio.item() if isinstance(neural_volume_ratio, torch.Tensor) else neural_volume_ratio,
            'volume_preservation_improvement_ratio': improvement_ratio
        }
        return volume_info

    def _compute_mesh_volume(self):
        """Calculate the total volume of the original undeformed mesh."""
        if not self.precomputed or self.detJ_all is None:
             print("Warning: Precomputed detJ not available for volume calculation.")
             return torch.tensor(float('nan'), device=self.device)
        quad_weights_r = self.quadrature_weights.unsqueeze(0)
        element_volumes = torch.sum(self.detJ_all * quad_weights_r, dim=1)
        total_volume = torch.sum(element_volumes)
        return torch.clamp(total_volume, min=0.0)

    def _compute_deformed_volume(self, displacement):
        """Calculate the total volume of the mesh after applying displacement."""
        if not self.precomputed: raise RuntimeError("Deformed volume requires precomputed matrices.")
        if displacement.dim() == 1: u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes: u_sample = displacement
        else: raise ValueError(f"Invalid displacement shape: {displacement.shape}")
        element_disps = u_sample[self.elements]
        total_volume = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for q_idx in range(self.num_quad_points):
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :]
            detJ_ref_q = self.detJ_all[:, q_idx]
            qw_q = self.quadrature_weights[q_idx]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u
            J_deformed = torch.linalg.det(F)
            total_volume += torch.sum(J_deformed * detJ_ref_q * qw_q)
        return total_volume

    def compute_deformation_gradients(self, displacement):
        """
        Compute the deformation gradients (F) for a given displacement field.

        Args:
            displacement: Displacement tensor [num_nodes*dim] or [num_nodes, dim]

        Returns:
            Deformation gradients F [num_elements, num_quad_points, dim, dim]
        """
        if not self.precomputed:
            raise RuntimeError("Deformation gradient computation requires precomputed matrices.")

        # Ensure displacement is in the correct shape [num_nodes, dim]
        if displacement.dim() == 1:
            u_sample = displacement.view(self.num_nodes, self.dim)
        elif displacement.dim() == 2 and displacement.shape[0] == self.num_nodes:
            u_sample = displacement
        else:
            raise ValueError(f"Invalid displacement shape: {displacement.shape}")

        # Get element displacements [num_elements, nodes_per_element, dim]
        element_disps = u_sample[self.elements]

        # Initialize deformation gradients tensor
        deformation_gradients = torch.zeros(
            (self.num_elements, self.num_quad_points, self.dim, self.dim),
            dtype=self.dtype, device=self.device
        )

        # Loop over quadrature points
        for q_idx in range(self.num_quad_points):
            # Get precomputed data for this quad point:
            dN_dx_q = self.dN_dx_all[:, q_idx, :, :]  # [num_elements, nodes_per_element, dim]

            # Compute grad(u) = ∑ u_n * dNn/dx : [num_elements, dim, dim]
            grad_u = torch.einsum('enj,enk->ejk', element_disps, dN_dx_q)

            # Compute deformation gradient F = I + grad(u) : [num_elements, dim, dim]
            I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
            F = I + grad_u

            # Store deformation gradients for this quadrature point
            deformation_gradients[:, q_idx, :, :] = F

        return deformation_gradients

import torch

class SOFAStVenantKirchhoffModelModified(torch.nn.Module):
    """
    PyTorch-based St. Venant-Kirchhoff (StVK) energy model.
    MODIFIED to optionally use a (J-1)^2 volumetric term for linear tetrahedra,
    similar to the first provided code snippet.

    Standard StVK: W = 0.5 * λ * (tr(E))² + μ * tr(E²)
    Modified form (if enabled): W_density = custom_lame * (J-1)² + μ * tr(E²)
    where E = 0.5 * (FᵀF - I), J = det(F).
    """

    def __init__(self, coordinates_np, elements_np, degree,
                 # Option 1: Standard E, nu
                 E=None, nu=None,
                 # Option 2: Direct Lame-like params for modified model
                 custom_lame_for_J_minus_1_sq=None, mu_direct=None,
                 use_J_minus_1_sq_volumetric=False, # Flag to switch model
                 precompute_matrices=True, device=None, dtype=torch.float64):
        super(SOFAStVenantKirchhoffModelModified, self).__init__()

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        print(f"SOFAStVenantKirchhoffModelModified: Using device: {self.device}, dtype: {self.dtype}")

        self.use_J_minus_1_sq_volumetric = use_J_minus_1_sq_volumetric

        if self.use_J_minus_1_sq_volumetric:
            if custom_lame_for_J_minus_1_sq is None or mu_direct is None:
                raise ValueError("For J-1 squared volumetric term, 'custom_lame_for_J_minus_1_sq' and 'mu_direct' must be provided.")
            self.custom_lame = torch.tensor(custom_lame_for_J_minus_1_sq, dtype=self.dtype, device=self.device)
            self.mu = torch.tensor(mu_direct, dtype=self.dtype, device=self.device)
            # lmbda is not used in this custom formulation
            self.lmbda = torch.tensor(0.0, dtype=self.dtype, device=self.device) # Placeholder
            print(f"Using MODIFIED StVK: custom_lame (for (J-1)^2)={self.custom_lame.item():.2f}, mu={self.mu.item():.2f}")
        else:
            if E is None or nu is None:
                raise ValueError("For standard StVK, 'E' and 'nu' must be provided.")
            self.E = torch.tensor(E, dtype=self.dtype, device=self.device)
            self.nu = torch.tensor(nu, dtype=self.dtype, device=self.device)
            if torch.abs(1.0 - 2.0 * self.nu) < 1e-9 or torch.abs(1.0 + self.nu) < 1e-9:
                 print("Warning: nu is close to 0.5 or -1. Adjusting Lamé parameters calculation.")
                 safe_nu = torch.clamp(self.nu, min=-0.99999, max=0.49999)
                 self.lmbda = self.E * safe_nu / ((1 + safe_nu) * (1 - 2 * safe_nu))
                 self.mu = self.E / (2 * (1 + safe_nu))
            else:
                 self.lmbda = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
                 self.mu = self.E / (2 * (1 + self.nu))
            print(f"Using STANDARD StVK: E={self.E.item():.2f}, nu={self.nu.item():.2f}, mu={self.mu.item():.2f}, lambda={self.lmbda.item():.2f}")


        self.degree = degree
        self.coordinates = torch.tensor(coordinates_np, device=self.device, dtype=self.dtype)
        self.elements = torch.tensor(elements_np, device=self.device, dtype=torch.long)

        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        self.num_elements = self.elements.shape[0]
        self.nodes_per_element = self.elements.shape[1]

        if self.dim != 3:
            print(f"Warning: Expected 3D coordinates, but got {self.dim}D.")
        if self.use_J_minus_1_sq_volumetric and self.nodes_per_element != 4:
            print(f"Warning: (J-1)^2 volumetric term is primarily intended/verified for linear tetrahedra (4 nodes). Current: {self.nodes_per_element} nodes.")


        print(f"Mesh info loaded: {self.num_nodes} nodes, {self.num_elements} elements")
        print(f"Nodes per element: {self.nodes_per_element} (degree={self.degree})")

        self._setup_elements(precompute_matrices)
        self.precompute_matrices = precompute_matrices
        self.eps = torch.tensor(1e-10, dtype=self.dtype, device=self.device)

    # --- _setup_elements, _generate_quadrature, _precompute_derivatives, _shape_function_derivatives_ref ---
    # These methods remain largely the same as in your original Code 2.
    # For linear tetrahedra, the existing 4-point quadrature is acceptable.
    # Since F is constant in a linear tet, the energy density will be constant
    # across these 4 QPs, and the sum of (Density * Vol_QP_i * Weight_QP_i)
    # will correctly give Density * Total_Vol_element.
    # If one strictly wanted to mimic a single-point evaluation like Code 1 might imply:
    # you could modify _generate_quadrature for nodes_per_element == 4 to use
    # 1 quadrature point (e.g., centroid [0.25,0.25,0.25]) and weight 1.0 (if detJ_all
    # is scaled to be the element volume directly, or 1/6 if detJ_all is det(dX/dxi_iso)).
    # However, the current 4-point rule is generally fine and more standard.

    def _setup_elements(self, precompute=True):
        """Setup element data and potentially precompute matrices"""
        self._generate_quadrature() # Depends on self.nodes_per_element

        if precompute:
            print("Precomputing element shape function derivatives...")
            self._precompute_derivatives()
            self.precomputed = True
            print("Precomputation finished.")
        else:
            print("Derivatives will be computed on-the-fly (less efficient).")
            self.precomputed = False
            self.dN_dx_all = None
            self.detJ_all = None # This stores det(dX/dxi_iso) for each element and QP

    def _generate_quadrature(self):
        """Generate quadrature rules based ONLY on nodes_per_element."""
        if self.nodes_per_element == 4:  # Linear Tetrahedron
            self.quadrature_points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.tensor([0.25, 0.25, 0.25, 0.25],
                                                  dtype=self.dtype,
                                                  device=self.device) / 6.0 # 1/6 for tet volume
            print(f"Using 4-point quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 8: # Linear Hexahedron
            gp = 1.0 / torch.sqrt(torch.tensor(3.0, device=self.device, dtype=self.dtype))
            self.quadrature_points = torch.tensor([
                [-gp, -gp, -gp], [ gp, -gp, -gp], [ gp,  gp, -gp], [-gp,  gp, -gp],
                [-gp, -gp,  gp], [ gp, -gp,  gp], [ gp,  gp,  gp], [-gp,  gp,  gp]
            ], dtype=self.dtype, device=self.device)
            self.quadrature_weights = torch.ones(8, dtype=self.dtype, device=self.device)
            print(f"Using 8-point (2x2x2 Gauss) quadrature for {self.nodes_per_element}-node elements.")
        elif self.nodes_per_element == 10: # Quadratic Tetrahedron
             print(f"Warning: Using placeholder 5-point quadrature for {self.nodes_per_element}-node Tet. Verify accuracy.")
             a = 0.108103018168070; b = 0.445948490915965
             self.quadrature_points = torch.tensor([
                 [0.25, 0.25, 0.25], [a, a, a], [b, a, a], [a, b, a], [a, a, b]
             ], dtype=self.dtype, device=self.device)
             w0 = -0.8 / 6.0; w1 = 0.325 / 6.0 # 1/6 for tet volume
             self.quadrature_weights = torch.tensor([w0, w1, w1, w1, w1], dtype=self.dtype, device=self.device) * 4.0 # Not sure about this *4.0
        else:
            raise NotImplementedError(f"Quadrature rule not implemented for {self.nodes_per_element}-node elements.")
        self.num_quad_points = len(self.quadrature_points)


    def _precompute_derivatives(self):
        num_qp = self.num_quad_points
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, self.dim),
                                    dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp), # This stores det(dX/dxi_iso)
                                   dtype=self.dtype, device=self.device)
        for e_idx in range(self.num_elements):
            element_node_indices = self.elements[e_idx]
            element_coords = self.coordinates[element_node_indices] # Rest coordinates X
            for q_idx, qp_ref in enumerate(self.quadrature_points):
                dN_dxi = self._shape_function_derivatives_ref(qp_ref) # Derivatives in ref element coords
                J_mat = torch.einsum('ni,nj->ij', element_coords, dN_dxi) # Jacobian of X(xi) map
                try:
                    detJ = torch.linalg.det(J_mat)
                    invJ = torch.linalg.inv(J_mat)
                except torch.linalg.LinAlgError as err:
                     print(f"Error computing inv/det(J_mat) for element {e_idx} at QP {q_idx}: {err}")
                     detJ = torch.tensor(0.0, dtype=self.dtype, device=self.device)
                     invJ = torch.zeros_like(J_mat)
                if detJ <= 0: # detJ is det(dX/dxi)
                     print(f"Warning: Non-positive Jacobian determinant ({detJ.item():.4e}) for element {e_idx} at QP {q_idx}. Check mesh quality.")
                dN_dX = torch.einsum('nj,jk->nk', dN_dxi, invJ) # dN/dX (derivatives w.r.t. material/rest coords)
                self.dN_dx_all[e_idx, q_idx] = dN_dX
                self.detJ_all[e_idx, q_idx] = detJ # Store det(dX/dxi_iso)

    def _shape_function_derivatives_ref(self, qp_ref):
        if self.nodes_per_element == 4: # Linear Tetrahedron
             dN_dxi = torch.tensor([
                 [-1.0, -1.0, -1.0], [ 1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0], [ 0.0,  0.0,  1.0]
             ], dtype=self.dtype, device=self.device)
        elif self.nodes_per_element == 8: # Linear Hexahedron
            xi, eta, zeta = qp_ref[0], qp_ref[1], qp_ref[2]
            one = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            xim, xip, etam, etap, zetam, zetap = one-xi, one+xi, one-eta, one+eta, one-zeta, one+zeta
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            eighth = 0.125
            dN_dxi[0, 0] = -eighth * etam * zetam; dN_dxi[1, 0] =  eighth * etam * zetam; dN_dxi[2, 0] =  eighth * etap * zetam; dN_dxi[3, 0] = -eighth * etap * zetam
            dN_dxi[4, 0] = -eighth * etam * zetap; dN_dxi[5, 0] =  eighth * etam * zetap; dN_dxi[6, 0] =  eighth * etap * zetap; dN_dxi[7, 0] = -eighth * etap * zetap
            dN_dxi[0, 1] = -eighth * xim * zetam; dN_dxi[1, 1] = -eighth * xip * zetam; dN_dxi[2, 1] =  eighth * xip * zetam; dN_dxi[3, 1] =  eighth * xim * zetam
            dN_dxi[4, 1] = -eighth * xim * zetap; dN_dxi[5, 1] = -eighth * xip * zetap; dN_dxi[6, 1] =  eighth * xip * zetap; dN_dxi[7, 1] =  eighth * xim * zetap
            dN_dxi[0, 2] = -eighth * xim * etam; dN_dxi[1, 2] = -eighth * xip * etam; dN_dxi[2, 2] = -eighth * xip * etap; dN_dxi[3, 2] = -eighth * xim * etap
            dN_dxi[4, 2] =  eighth * xim * etam; dN_dxi[5, 2] =  eighth * xip * etam; dN_dxi[6, 2] =  eighth * xip * etap; dN_dxi[7, 2] =  eighth * xim * etap
        else:
            raise NotImplementedError(f"Shape function derivatives not implemented for {self.nodes_per_element}-node elements.")
        return dN_dxi

    def forward(self, u_tensor):
        return self.compute_energy(u_tensor)

    def compute_energy(self, displacement_batch):
        if not self.precomputed:
             raise RuntimeError("Energy computation requires precomputed matrices.")

        is_batch = displacement_batch.dim() > 1
        if not is_batch:
            displacement_batch = displacement_batch.unsqueeze(0)
        batch_size = displacement_batch.shape[0]
        # u_reshaped should be [batch_size, num_total_nodes, dim]
        # Ensure displacement_batch is flattened per sample then reshaped
        u_reshaped = displacement_batch.view(batch_size, self.num_nodes, self.dim)


        total_energy = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        for b in range(batch_size):
            u_sample = u_reshaped[b] # [num_nodes, dim]
            # element_disps: [num_elements, nodes_per_element, dim]
            element_disps = u_sample[self.elements]
            energy_sample = torch.tensor(0.0, dtype=self.dtype, device=self.device)

            for q_idx in range(self.num_quad_points):
                # dN_dX_q: [num_elements, nodes_per_element, dim] (derivatives w.r.t. material coords X)
                dN_dX_q = self.dN_dx_all[:, q_idx, :, :]
                # detJ_iso_q: [num_elements] (Jacobian determinant dX/dxi_iso)
                detJ_iso_q = self.detJ_all[:, q_idx]
                # qw_q: scalar (quadrature weight for this point)
                qw_q = self.quadrature_weights[q_idx]

                # grad_u = sum_nodes (u_node * dN_node/dX)
                # element_disps: [E, N_per_E, D]
                # dN_dX_q:       [E, N_per_E, D]
                # grad_u:        [E, D, D]
                grad_u = torch.einsum('end,enk->edk', element_disps, dN_dX_q) # Corrected einsum for grad_u
                I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(self.num_elements, -1, -1)
                F = I + grad_u # Deformation Gradient F_ij = dx_i/dX_j

                energy_density_q = self._compute_energy_density_at_qp(F) # [num_elements]

                # dV0 = det(dX/dxi_iso) * quad_weight_iso
                # For tets, quad_weight_iso includes the 1/6 factor.
                # So, detJ_iso_q * qw_q is the rest volume differential dV0 for this QP.
                element_volume_differential = detJ_iso_q * qw_q
                energy_sample += torch.sum(energy_density_q * element_volume_differential)

            total_energy[b] = energy_sample

        if not is_batch:
            total_energy = total_energy.squeeze(0)
        return total_energy


    def _compute_energy_density_at_qp(self, F):
        """
        Compute St. Venant-Kirchhoff (StVK) strain energy density.
        Can use standard or modified ((J-1)^2) volumetric term.
        """
        if F.shape[-2:] != (self.dim, self.dim):
             raise ValueError(f"Input F must have shape [..., {self.dim}, {self.dim}], but got {F.shape}")

        batch_dims = F.shape[:-2] # E.g., [num_elements]
        I = torch.eye(self.dim, dtype=self.dtype, device=self.device).expand(*batch_dims, -1, -1)

        C = torch.einsum('...ji,...jk->...ik', F, F) # Right Cauchy-Green C = FᵀF
        E = 0.5 * (C - I) # Green-Lagrange strain E

        # Deviatoric part (common to both formulations)
        # tr(E²) = ∑_i,j (E_ij * E_ij) [*]
        trE2 = torch.einsum('...ij,...ij->...', E, E)
        deviatoric_energy_density = self.mu * trE2

        # Volumetric part
        if self.use_J_minus_1_sq_volumetric:
            J = torch.linalg.det(F) # Determinant of F [*]
            volumetric_energy_density = self.custom_lame * (J - 1.0) ** 2
        else: # Standard StVK
            trE = torch.sum(torch.diagonal(E, dim1=-2, dim2=-1), dim=-1) # tr(E) [*]
            volumetric_energy_density = 0.5 * self.lmbda * (trE ** 2)

        W_density = volumetric_energy_density + deviatoric_energy_density
        return W_density

    # --- Other methods (compute_gradient, compute_PK1, etc.) would also need to be
    #     consistent with the chosen energy formulation if used.
    #     For simplicity, they are omitted here but would need careful review. ---

    def compute_gradient(self, displacement_batch):
        """
        Compute internal forces (negative gradient of energy w.r.t. displacements).
        Uses torch.autograd for automatic differentiation.
        """
        # Ensure grad is enabled for this computation if called from a context where it might be disabled
        is_grad_enabled_globally = torch.is_grad_enabled()
        if not is_grad_enabled_globally:
            torch.set_grad_enabled(True)

        if not displacement_batch.requires_grad:
            # If called multiple times, ensure it's a fresh tensor for grad computation
            # or handle appropriately based on expected use.
            # For a simple call, detaching and cloning is safest.
            displacement_batch_for_grad = displacement_batch.detach().clone().requires_grad_(True)
        else:
            displacement_batch_for_grad = displacement_batch

        energy = self.compute_energy(displacement_batch_for_grad)
        grad_outputs = torch.ones_like(energy) # For batched energy

        # Summing energy for scalar output if batching, or if energy is already scalar
        energy_sum = energy.sum()

        grad = torch.autograd.grad(
            outputs=energy_sum,
            inputs=displacement_batch_for_grad,
            grad_outputs=None, # Since energy_sum is scalar
            create_graph=torch.is_grad_enabled(), # Propagate graph if nested diff
            retain_graph=True # Often needed if grad is used for further ops or multiple grad calls
        )[0]

        if not is_grad_enabled_globally:
            torch.set_grad_enabled(False) # Restore global state

        return grad




# --- Boundary Condition Managers (from your example, can be kept separate) ---
class BoundaryConditionManager:
    """Manages boundary conditions for FEM problems"""
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=torch.float, device=self.device)

    def set_fixed_dofs(self, indices, values):
        self.fixed_dofs = indices.to(self.device) if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long, device=self.device)
        self.fixed_values = values.to(self.device) if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float, device=self.device)

    def apply(self, displacement_batch):
        if self.fixed_dofs.numel() == 0:
            return displacement_batch
        u_batch_fixed = displacement_batch.clone()
        batch_size = displacement_batch.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.fixed_dofs.numel())
        flat_batch_indices = batch_indices.reshape(-1)
        repeated_dofs = self.fixed_dofs.repeat(batch_size)
        u_batch_fixed[flat_batch_indices, repeated_dofs] = self.fixed_values.repeat(batch_size)
        return u_batch_fixed

class SmoothBoundaryConditionManager:
    """Manages boundary conditions for FEM problems with smooth enforcement via penalty"""
    def __init__(self, device=None, penalty_strength=1e3):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fixed_dofs = torch.tensor([], dtype=torch.long, device=self.device)
        self.fixed_values = torch.tensor([], dtype=torch.float, device=self.device)
        self.penalty_strength = penalty_strength

    def set_fixed_dofs(self, indices, values):
        self.fixed_dofs = indices.to(self.device) if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long, device=self.device)
        self.fixed_values = values.to(self.device) if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=torch.float, device=self.device)

    def apply(self, displacement_batch):
        return displacement_batch

    def compute_penalty_energy(self, displacement_batch):
        if self.fixed_dofs.numel() == 0:
            return torch.zeros(displacement_batch.shape[0], device=self.device, dtype=displacement_batch.dtype)
        batch_size = displacement_batch.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, self.fixed_dofs.numel())
        flat_batch_indices = batch_indices.reshape(-1)
        repeated_dofs = self.fixed_dofs.repeat(batch_size)
        repeated_values = self.fixed_values.repeat(batch_size)
        actual_values = displacement_batch[flat_batch_indices, repeated_dofs]
        squared_diff = torch.pow(actual_values - repeated_values, 2)
        squared_diff = squared_diff.reshape(batch_size, -1)
        penalty_energy = self.penalty_strength * squared_diff.sum(dim=1)
        return penalty_energy

# ==============================================================================
# Custom Autograd Function for Implicit Differentiation (No Visualization)
# ==============================================================================
class FEMSolverFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f_ext, solver_instance): # Removed disp_gt_subset
        if not f_ext.requires_grad:
            with torch.no_grad():
                 f_ext_detached = f_ext.clone()
                 u_star = solver_instance._solve_with_lbfgs_internal(f_ext_detached, u0=None)
            return u_star
        else:
            f_ext_detached = f_ext.detach().clone()
            u_star = solver_instance._solve_with_lbfgs_internal(f_ext_detached, u0=None)
            ctx.solver_instance = solver_instance
            ctx.save_for_backward(u_star, f_ext_detached)
            return u_star

    @staticmethod
    def backward(ctx, grad_output):
        solver_instance = ctx.solver_instance
        u_star_saved, f_ext_detached = ctx.saved_tensors

        if grad_output is None:
            return None, None # Adjusted for removed disp_gt_subset

        batch_size = grad_output.shape[0]
        adjoint_solutions = []

        def hvp_at_ustar(v, u_star_current_batch):
            v_detached = v.detach()
            with torch.enable_grad():
                u_star_input = u_star_current_batch.detach().clone().requires_grad_(True)
                f_int_total_for_hvp = solver_instance.energy_model.compute_gradient(u_star_input)
            hvp_result, = torch.autograd.grad(
                outputs=f_int_total_for_hvp,
                inputs=u_star_input,
                grad_outputs=v_detached,
                retain_graph=False,
                create_graph=False
            )
            return hvp_result

        for i in range(batch_size):
            grad_output_i = grad_output[i:i+1]
            u_star_i = u_star_saved[i:i+1]
            hvp_func_for_cg = lambda vec: hvp_at_ustar(vec, u_star_i)
            v = solver_instance._solve_linear_system_cg(
                hvp_function=hvp_func_for_cg,
                rhs=grad_output_i,
                max_iter=solver_instance.cg_max_iter_backward,
                tol=solver_instance.cg_tol_backward
            )
            adjoint_solutions.append(v)
        grad_f_ext = torch.cat(adjoint_solutions, dim=0)
        return grad_f_ext, None # Adjusted for removed disp_gt_subset

# ==============================================================================
# Differentiable Solver Class (L-BFGS Forward, No Visualization)
# ==============================================================================
class DifferentiableFEMSolverIFT(torch.nn.Module):
    def __init__(self, energy_model: EnergyModel,
                 lbfgs_max_iter=50, lbfgs_tolerance_grad=1e-5, lbfgs_tolerance_change=1e-7, lbfgs_lr=1.0, lbfgs_history_size=100,
                 cg_max_iter_backward=300, cg_tol_backward=1e-6,
                 verbose=False):
        super().__init__()
        self.energy_model = energy_model
        self.verbose = verbose
        self.device = energy_model.device
        self.dtype = energy_model.dtype

        if not all(hasattr(energy_model, attr) for attr in ['num_nodes', 'dim', 'device', 'dtype', 'compute_gradient', 'compute_energy']):
             raise AttributeError("energy_model must have 'num_nodes', 'dim', 'device', 'dtype', 'compute_gradient', 'compute_energy' attributes/methods.")
        self.num_nodes = energy_model.num_nodes
        self.dim = energy_model.dim
        self.dof_count = self.num_nodes * self.dim

        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tolerance_grad = lbfgs_tolerance_grad
        self.lbfgs_tolerance_change = lbfgs_tolerance_change
        self.lbfgs_lr = lbfgs_lr
        self.lbfgs_history_size = lbfgs_history_size

        self.cg_max_iter_backward = cg_max_iter_backward
        self.cg_tol_backward = cg_tol_backward

    def forward(self, external_forces): # Removed disp_gt_subset
        return FEMSolverFunction.apply(external_forces, self) # Pass self as solver_instance

    def _solve_with_lbfgs_internal(self, external_forces_detached, u0=None): # Removed disp_gt_subset_for_viz
        batch_size = external_forces_detached.shape[0]
        solutions = []

        for i in range(batch_size):
            if self.verbose and batch_size > 1: print(f"\n--- LBFGS Solve: Sample {i+1}/{batch_size} ---")
            f_i_detached = external_forces_detached[i:i+1]

            if u0 is None:
                current_u_i = torch.randn_like(f_i_detached) * 1e-5
            else:
                current_u_i = u0[i:i+1].detach().clone()
            current_u_i.requires_grad_(True)

            optimizer = torch.optim.LBFGS([current_u_i],
                                    lr=self.lbfgs_lr,
                                    max_iter=self.lbfgs_max_iter,
                                    history_size=self.lbfgs_history_size,
                                    tolerance_grad=self.lbfgs_tolerance_grad,
                                    tolerance_change=self.lbfgs_tolerance_change,
                                    line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                internal_energy_val = self.energy_model.compute_energy(current_u_i)
                external_work_val = torch.sum(f_i_detached * current_u_i)
                total_potential_energy_val = internal_energy_val - external_work_val
                total_potential_energy_val.backward()
                return total_potential_energy_val
            try:
                optimizer.step(closure)
            except Exception as e:
                print(f"LBFGS optimization failed for sample {i}: {e}")
                solutions.append(current_u_i.detach().clone())
                continue

            final_loss = closure() # Re-evaluate to get final state
            if self.verbose:
                grad_norm = current_u_i.grad.norm().item() if current_u_i.grad is not None else float('nan')
                print(f"Sample {i}: LBFGS finished. Final Potential Energy = {final_loss.item():.3e}, Grad Norm = {grad_norm:.3e}")
            solutions.append(current_u_i.detach().clone())

        if not solutions:
            return torch.empty((0, self.dof_count), device=self.device, dtype=self.dtype)
        final_solutions = torch.cat(solutions, dim=0)
        return final_solutions

    def _solve_linear_system_cg(self, hvp_function, rhs, max_iter, tol):
        with torch.no_grad():
            x = torch.zeros_like(rhs)
            r = rhs.clone()
            p = r.clone()
            rsold_sq = torch.dot(r.flatten(), r.flatten())
            rsinit_norm = torch.sqrt(rsold_sq).item()
            if self.verbose and self.verbose > 1 : print(f"CG Start: Initial ||Residual|| = {rsinit_norm:.3e}, Max Iter = {max_iter}, Tol = {tol:.1e}")

            if rsinit_norm < 1e-15:
                if self.verbose and self.verbose > 1: print("CG: Initial residual near zero. Returning zero solution.")
                return x
            best_x = x.clone()
            min_residual_norm_sq = rsold_sq.item()

            for i_cg in range(max_iter):
                try:
                    Ap = hvp_function(p)
                except Exception as e:
                    print(f"CG Error: HVP computation failed at iteration {i_cg}: {e}")
                    return best_x
                if torch.isnan(Ap).any() or torch.isinf(Ap).any():
                    print(f"CG Error: NaN/Inf in HVP result at iter {i_cg}.")
                    return best_x
                pAp_dot = torch.dot(p.flatten(), Ap.flatten())
                if pAp_dot <= 1e-12 * torch.dot(p.flatten(), p.flatten()):
                    if self.verbose and self.verbose > 1: print(f"CG Warning: Breakdown at iter {i_cg} (p^T*A*p = {pAp_dot.item():.3e}). Returning best solution.")
                    return best_x
                alpha = rsold_sq / pAp_dot
                x.add_(p, alpha=alpha)
                r.add_(Ap, alpha=-alpha)
                rsnew_sq = torch.dot(r.flatten(), r.flatten())
                current_residual_norm = torch.sqrt(rsnew_sq).item()
                if rsnew_sq.item() < min_residual_norm_sq:
                    min_residual_norm_sq = rsnew_sq.item()
                    best_x = x.clone()
                if current_residual_norm < tol * rsinit_norm:
                    if self.verbose and self.verbose > 1: print(f"CG Converged at iteration {i_cg+1}. Final Rel Residual = {current_residual_norm / rsinit_norm:.3e}")
                    break
                beta = rsnew_sq / rsold_sq
                p = r + beta * p
                rsold_sq = rsnew_sq
                if self.verbose and self.verbose > 1 and (i_cg + 1) % 50 == 0:
                     print(f"CG Iter {i_cg+1}: Rel Residual = {current_residual_norm / rsinit_norm:.3e}")
            else:
                if self.verbose and self.verbose > 1: print(f"CG Warning: Max iterations ({max_iter}) reached. Final Rel Residual = {current_residual_norm / rsinit_norm:.3e}")
                x = best_x
            if self.verbose and self.verbose > 1: print(f"CG End: ||Solution|| = {torch.linalg.norm(x).item():.3e}")
            return x

class PabloNeoHookeanEnergy(torch.nn.Module):
    """
    Neo-Hookean energy implementation with Pablo's formulation, fully optimized
    using batch processing and vectorized operations for improved performance.
    """
    def __init__(self, domain, degree, E, nu, precompute_matrices=True, device=None, 
                 dtype=torch.float64, batch_size=100):
        super(PabloNeoHookeanEnergy, self).__init__()
        
        # Set device and precision
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.batch_size = batch_size
        
        # Material properties
        self.E = E
        self.nu = nu
        self.mu = E / (2 * (1 + nu))  # Shear modulus
        self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))  # First Lamé parameter
        
        # Extract mesh information
        self.coordinates = torch.tensor(domain.geometry.x, dtype=self.dtype, device=self.device)
        self.num_nodes = self.coordinates.shape[0]
        self.dim = self.coordinates.shape[1]
        
        # Create elements tensor
        elements_list = []
        tdim = domain.topology.dim
        for cell in range(domain.topology.index_map(tdim).size_local):
            elements_list.append(domain.topology.connectivity(tdim, 0).links(cell))
        
        # Convert directly to device tensor in one step
        self.elements = torch.tensor(np.array(elements_list), dtype=torch.long, device=self.device)
        self.num_elements = len(self.elements)
        self.nodes_per_element = self.elements.shape[1]
        
        # Determine element type
        if self.nodes_per_element == 4:
            self.element_type = "tetrahedron"
        elif self.nodes_per_element == 8:
            self.element_type = "hexahedron"
        else:
            raise ValueError(f"Unsupported element type with {self.nodes_per_element} nodes")
        
        print(f"Mesh: {self.num_elements} {self.element_type} elements, {self.num_nodes} nodes")
        
        # Generate quadrature points - optimized for Neo-Hookean materials
        self.quadrature_points, self.quadrature_weights = self._generate_quadrature()
        print(f"Using {len(self.quadrature_weights)} quadrature points per element")
        
        # Memory-efficient precomputation strategy
        self.precomputed = False
        if precompute_matrices:
            self._precompute_derivatives_in_chunks()
            self.precomputed = True
            print("Precomputation complete")

    def _estimate_precompute_memory(self):
        """Estimate memory needed for precomputation in GB"""
        num_qp = len(self.quadrature_points)
        bytes_per_element = (
            # dN_dx_all: [num_elements, num_qp, nodes_per_element, 3]
            self.num_elements * num_qp * self.nodes_per_element * 3 * self.dtype.itemsize +
            # detJ_all: [num_elements, num_qp]
            self.num_elements * num_qp * self.dtype.itemsize
        )
        return bytes_per_element / (1024**3)  # Convert to GB

    def _generate_quadrature(self):
        """Generate quadrature rules optimized for Neo-Hookean materials"""
        if self.element_type == "tetrahedron":
            # 4-point quadrature for tetrahedron (more accurate for nonlinear materials)
            points = torch.tensor([
                [0.58541020, 0.13819660, 0.13819660],
                [0.13819660, 0.58541020, 0.13819660],
                [0.13819660, 0.13819660, 0.58541020],
                [0.13819660, 0.13819660, 0.13819660]
            ], dtype=self.dtype, device=self.device)
            weights = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=self.dtype, device=self.device) / 6.0
        else:  # hexahedron
            # 2×2×2 Gaussian quadrature
            gp = 1.0 / math.sqrt(3)
            points = []
            weights = []
            for i in [-gp, gp]:
                for j in [-gp, gp]:
                    for k in [-gp, gp]:
                        points.append([i, j, k])
                        weights.append(1.0)
            points = torch.tensor(points, dtype=self.dtype, device=self.device)
            weights = torch.tensor(weights, dtype=self.dtype, device=self.device)
        
        return points, weights
    
    def _precompute_derivatives_in_chunks(self):
        """Precompute derivatives in a format optimized for batch operations"""
        print("Precomputing derivatives for efficient batch processing...")
        
        # Count total quadrature points per element
        num_qp = len(self.quadrature_points)
        
        # Initialize storage as tensors rather than lists of lists
        self.dN_dx_all = torch.zeros((self.num_elements, num_qp, self.nodes_per_element, 3), 
                                   dtype=self.dtype, device=self.device)
        self.detJ_all = torch.zeros((self.num_elements, num_qp), 
                                  dtype=self.dtype, device=self.device)
        
        # Process elements in chunks
        for chunk_start in range(0, self.num_elements, self.batch_size):
            chunk_end = min(chunk_start + self.batch_size, self.num_elements)
            
            for e in range(chunk_start, chunk_end):
                element_nodes = self.elements[e].long()
                element_coords = self.coordinates[element_nodes]
                
                for q_idx in range(num_qp):
                    qp = self.quadrature_points[q_idx]
                    dN_dx, detJ = self._compute_derivatives(element_coords, qp)
                    
                    # Store in tensor format
                    self.dN_dx_all[e, q_idx] = dN_dx
                    self.detJ_all[e, q_idx] = detJ
            
            # Clear GPU cache after each chunk
            if torch.cuda.is_available() and (chunk_end - chunk_start) > 1000:
                torch.cuda.empty_cache()
        
        est_memory = self._estimate_precompute_memory()
        print(f"Precomputation complete. Memory usage: {est_memory:.2f} GB")


    def _compute_derivatives_batch(self, batch_coords, qp):
        """
        Vectorized computation of shape function derivatives for batches of elements
        at a quadrature point.
        
        Args:
            batch_coords: Element coordinates [chunk_size, batch_size, nodes_per_element, 3]
            qp: Quadrature point coordinates [3]
            
        Returns:
            batch_dN_dx: Shape function derivatives [chunk_size, batch_size, nodes_per_element, 3]
            batch_detJ: Jacobian determinants [chunk_size, batch_size]
        """
        chunk_size, batch_size = batch_coords.shape[0], batch_coords.shape[1]
        
        if self.nodes_per_element == 4:  # tetrahedron
            # Shape function derivatives for tetrahedron (constant)
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=self.dtype, device=self.device)
            
            # Expand for all elements in batch
            batch_dN_dxi = dN_dxi.expand(chunk_size, batch_size, 4, 3)
            
            # Jacobian calculation using batch matrix multiplication
            # Reshape batch_coords to [chunk_size*batch_size, 4, 3]
            coords_reshaped = batch_coords.reshape(-1, 4, 3)
            
            # Compute Jacobians for all elements [chunk_size*batch_size, 3, 3]
            J_flat = torch.bmm(coords_reshaped.transpose(1, 2), 
                            dN_dxi.expand(chunk_size*batch_size, 4, 3))
            
            # Compute determinants and inverses
            detJ_flat = torch.linalg.det(J_flat)
            invJ_flat = torch.linalg.inv(J_flat)
            
            # Reshape back to batch dimensions
            batch_detJ = detJ_flat.reshape(chunk_size, batch_size)
            batch_invJ = invJ_flat.reshape(chunk_size, batch_size, 3, 3)
            
            # Compute derivatives w.r.t. physical coordinates using batched operation
            # For each chunk and batch element, multiply dN_dxi by invJ
            batch_dN_dx = torch.zeros((chunk_size, batch_size, 4, 3), 
                                    dtype=self.dtype, device=self.device)
            
            for c in range(chunk_size):
                for b in range(batch_size):
                    batch_dN_dx[c, b] = torch.matmul(dN_dxi, batch_invJ[c, b])
            
        else:  # hexahedron
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim, xip = 1.0 - xi, 1.0 + xi
            etam, etap = 1.0 - eta, 1.0 + eta
            zetam, zetap = 1.0 - zeta, 1.0 + zeta
            
            # Create shape function derivatives tensor for hexahedron
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=self.device)
            
            # First derivatives with respect to xi
            dN_dxi[:, 0] = torch.tensor([
                -0.125 * etam * zetam, 0.125 * etam * zetam, 
                0.125 * etap * zetam, -0.125 * etap * zetam,
                -0.125 * etam * zetap, 0.125 * etam * zetap,
                0.125 * etap * zetap, -0.125 * etap * zetap
            ], dtype=self.dtype, device=self.device)
            
            # First derivatives with respect to eta
            dN_dxi[:, 1] = torch.tensor([
                -0.125 * xim * zetam, -0.125 * xip * zetam,
                0.125 * xip * zetam, 0.125 * xim * zetam,
                -0.125 * xim * zetap, -0.125 * xip * zetap,
                0.125 * xip * zetap, 0.125 * xim * zetap
            ], dtype=self.dtype, device=self.device)
            
            # First derivatives with respect to zeta
            dN_dxi[:, 2] = torch.tensor([
                -0.125 * xim * etam, -0.125 * xip * etam,
                -0.125 * xip * etap, -0.125 * xim * etap,
                0.125 * xim * etam, 0.125 * xip * etam,
                0.125 * xip * etap, 0.125 * xim * etap
            ], dtype=self.dtype, device=self.device)
            
            # Expand for all elements in batch
            batch_dN_dxi = dN_dxi.expand(chunk_size, batch_size, 8, 3)
            
            # Process in smaller batches to avoid memory issues
            max_inner_batch = 128
            batch_detJ = torch.zeros((chunk_size, batch_size), dtype=self.dtype, device=self.device)
            batch_dN_dx = torch.zeros((chunk_size, batch_size, 8, 3), dtype=self.dtype, device=self.device)
            
            for cb in range(0, chunk_size*batch_size, max_inner_batch):
                end_idx = min(cb + max_inner_batch, chunk_size*batch_size)
                current_batch = end_idx - cb
                
                # Convert flat indices to chunk, batch indices
                c_indices = [cb // batch_size + i // batch_size for i in range(current_batch)]
                b_indices = [cb % batch_size + i % batch_size for i in range(current_batch)]
                
                # Extract coordinates for current batch
                coords_batch = torch.stack([batch_coords[c, b] for c, b in zip(c_indices, b_indices)])
                
                # Compute Jacobians
                J_batch = torch.bmm(coords_batch.transpose(1, 2), 
                                dN_dxi.expand(current_batch, 8, 3))
                
                # Compute determinants and inverses
                detJ_batch = torch.linalg.det(J_batch)
                invJ_batch = torch.linalg.inv(J_batch)
                
                # Compute derivatives
                dN_dx_batch = torch.bmm(dN_dxi.expand(current_batch, 8, 3), invJ_batch)
                
                # Store results back in original tensors
                for i, (c, b) in enumerate(zip(c_indices, b_indices)):
                    batch_detJ[c, b] = detJ_batch[i]
                    batch_dN_dx[c, b] = dN_dx_batch[i]
        
        return batch_dN_dx, batch_detJ
    
    def _compute_derivatives(self, element_coords, qp):
        """Compute shape function derivatives for an element at a quadrature point"""
        if element_coords.shape[0] == 4:  # tetrahedron
            # Shape function derivatives for tetrahedron (constant)
            dN_dxi = torch.tensor([
                [-1.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        else:  # hexahedron with 8 nodes
            xi, eta, zeta = qp
            
            # Precompute terms for efficiency
            xim = 1.0 - xi
            xip = 1.0 + xi
            etam = 1.0 - eta
            etap = 1.0 + eta
            zetam = 1.0 - zeta
            zetap = 1.0 + zeta
            
            # Shape function derivatives using vectorized operations
            dN_dxi = torch.zeros((8, 3), dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to xi
            dN_dxi[:, 0] = torch.tensor([
                -0.125 * etam * zetam,
                0.125 * etam * zetam,
                0.125 * etap * zetam,
                -0.125 * etap * zetam,
                -0.125 * etam * zetap,
                0.125 * etam * zetap,
                0.125 * etap * zetap,
                -0.125 * etap * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to eta
            dN_dxi[:, 1] = torch.tensor([
                -0.125 * xim * zetam,
                -0.125 * xip * zetam,
                0.125 * xip * zetam,
                0.125 * xim * zetam,
                -0.125 * xim * zetap,
                -0.125 * xip * zetap,
                0.125 * xip * zetap,
                0.125 * xim * zetap
            ], dtype=self.dtype, device=element_coords.device)
            
            # First derivatives with respect to zeta
            dN_dxi[:, 2] = torch.tensor([
                -0.125 * xim * etam,
                -0.125 * xip * etam,
                -0.125 * xip * etap,
                -0.125 * xim * etap,
                0.125 * xim * etam,
                0.125 * xip * etam,
                0.125 * xip * etap,
                0.125 * xim * etap
            ], dtype=self.dtype, device=element_coords.device)
            
            # Jacobian calculation
            J = torch.matmul(element_coords.T, dN_dxi)
            detJ = torch.det(J)
            invJ = torch.linalg.inv(J)
            
            # Shape function derivatives w.r.t. physical coordinates
            dN_dx = torch.matmul(dN_dxi, invJ)
            
        return dN_dx, detJ
    
    def _compute_F_batch(self, batch_coords, batch_disps, batch_dN_dx):
        """
        Compute deformation gradient F = I + ∇u for a batch of elements using broadcasting
        
        Args:
            batch_coords: Element coordinates [batch_size, nodes_per_element, 3]
            batch_disps: Element displacements [batch_size, nodes_per_element, 3]
            batch_dN_dx: Shape function derivatives [batch_size, nodes_per_element, 3]
            
        Returns:
            Batch of deformation gradients [batch_size, 3, 3]
        """
        # Initialize deformation gradient as identity for each element in batch
        batch_size = batch_disps.shape[0]
        batch_F = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3).clone()
        
        # Use einsum for efficient batch computation: F += u_i ⊗ ∇N_i
        batch_F += torch.einsum('bij,bik->bjk', batch_disps, batch_dN_dx)
        
        return batch_F
    
    def _compute_F_batch_multi(self, batch_coords, batch_disps, batch_dN_dx):
        """
        Compute deformation gradient F = I + ∇u for a batch of elements across multiple samples
        
        Args:
            batch_coords: Element coordinates [chunk_size, batch_size, nodes_per_element, 3]
            batch_disps: Element displacements [chunk_size, batch_size, nodes_per_element, 3]
            batch_dN_dx: Shape function derivatives [chunk_size, batch_size, nodes_per_element, 3]
            
        Returns:
            Batch of deformation gradients [chunk_size, batch_size, 3, 3]
        """
        chunk_size, local_batch_size = batch_disps.shape[0], batch_disps.shape[1]
        
        # Pre-allocate the F tensor directly (more efficient than expanding identity)
        batch_F = torch.zeros((chunk_size, local_batch_size, 3, 3), 
                            dtype=self.dtype, device=self.device)
        
        # Set diagonal elements to 1 (identity matrix)
        batch_F[:, :, 0, 0] = 1.0
        batch_F[:, :, 1, 1] = 1.0
        batch_F[:, :, 2, 2] = 1.0
        
        # Add displacement gradient using einsum
        # This computes F = I + du/dX efficiently across all dimensions
        batch_F += torch.einsum('cbij,cbik->cbjk', batch_disps, batch_dN_dx)
        
        return batch_F
    
    def _compute_PK1_batch_multi(self, batch_F):
        """
        Compute the First Piola-Kirchhoff stress tensor for multiple batches of deformation gradients
        with improved numerical stability and memory efficiency.
        
        Args:
            batch_F: Tensor of shape [chunk_size, batch_size, 3, 3] containing deformation gradients
            
        Returns:
            Tensor of shape [chunk_size, batch_size, 3, 3] containing PK1 tensors
        """
        # Get batch dimensions for broadcasting
        chunk_size, batch_size = batch_F.shape[0], batch_F.shape[1]
        
        # Compute J = det(F) efficiently
        batch_J = torch.linalg.det(batch_F)
        
        # Check for problematic determinants (too small or negative)
        det_issues = (batch_J < 1e-10)
        if torch.any(det_issues):
            # Create a safe copy to avoid modifying the input tensor
            safe_F = batch_F.clone()
            
            # For problematic elements, modify F slightly to ensure invertibility
            # by adding a small identity component
            problematic_indices = torch.nonzero(det_issues, as_tuple=True)
            identity = torch.eye(3, device=batch_F.device, dtype=batch_F.dtype)
            for c, b in zip(*problematic_indices):
                # Add scaled identity based on F magnitude
                scale = torch.abs(batch_F[c, b]).mean() * 1e-4
                safe_F[c, b] = batch_F[c, b] + identity * scale
                
            # Recompute determinants for corrected F values
            batch_J = torch.where(det_issues, torch.linalg.det(safe_F), batch_J)
            
            # Use corrected F for inverse calculation
            batch_inv_F = torch.linalg.inv(safe_F)
        else:
            # No issues detected, proceed normally
            batch_inv_F = torch.linalg.inv(batch_F)
        
        # Compute F^-T (transpose the inverse)
        batch_inv_F_T = batch_inv_F.transpose(2, 3)
        
        # Precompute J^2-1 and reshape for broadcasting
        # This is more memory-efficient than using unsqueeze twice
        J_term = (batch_J * batch_J - 1.0).reshape(chunk_size, batch_size, 1, 1)
        
        # First Piola-Kirchhoff stress tensor computation using fused operations
        # Initialize with first term to avoid extra allocation
        batch_P = self.mu * batch_F
        
        # Subtract second term in-place
        batch_P.sub_(self.mu * batch_inv_F_T)
        
        # Add third term in-place
        batch_P.add_(0.5 * self.lmbda * J_term * batch_inv_F_T)
        
        return batch_P
        
    def _compute_neohook_energy_density_batch(self, batch_F):
        """
        Compute Neo-Hookean strain energy density for a batch of deformation gradients
        with improved stability handling based on Pablo's formulation
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
        
        Returns:
            Tensor of shape [batch_size] containing energy densities
        """
        # Compute J = det(F) for all elements in batch
        batch_J = torch.linalg.det(batch_F)
        
        # Compute right Cauchy-Green tensor and its first invariant
        batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
        batch_I1 = torch.diagonal(batch_C, dim1=1, dim2=2).sum(dim=1)
        
        # Create safe versions of variables for numerical stability
        # Following Pablo's implementation but with improved stability
        batch_size = batch_F.shape[0]
        batch_W = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # Split computation based on J values for stability
        # Case 1: Valid J values (J > threshold)
        valid_mask = batch_J > 1e-6
        if torch.any(valid_mask):
            valid_J = batch_J[valid_mask]
            valid_I1 = batch_I1[valid_mask]
            valid_log_J = torch.log(valid_J)
            
            # Calculate both terms exactly as in pablo_loss.py
            isochoric_term = 0.5 * self.mu * (valid_I1 - 3.0 - 2.0 * valid_log_J)
            volumetric_term = 0.25 * self.lmbda * ((valid_J*valid_J) - 1.0 - 2.0 * valid_log_J)
            
            # Total energy
            batch_W[valid_mask] = isochoric_term + volumetric_term
        
        # Case 2: Small positive J values (need stabilization)
        small_mask = (batch_J > 0) & (batch_J <= 1e-6)
        if torch.any(small_mask):
            small_J = batch_J[small_mask]
            small_I1 = batch_I1[small_mask]
            
            # Use Taylor expansion for log(J) when J is very small
            # log(J) ≈ J-1 - (J-1)²/2 + (J-1)³/3 - ...
            approx_log_J = small_J - 1.0 - (small_J - 1.0)**2 / 2.0
            
            isochoric_term = 0.5 * self.mu * (small_I1 - 3.0 - 2.0 * approx_log_J)
            volumetric_term = 0.25 * self.lmbda * ((small_J*small_J) - 1.0 - 2.0 * approx_log_J)
            
            # Add regularization to prevent energy from going negative
            barrier = 1e2 * torch.pow(1e-6 - small_J, 2)
            batch_W[small_mask] = isochoric_term + volumetric_term + barrier
        
        # Case 3: Negative J values (inverted elements)
        invalid_mask = batch_J <= 0
        if torch.any(invalid_mask):
            # Use a large positive penalty proportional to how negative J is
            batch_W[invalid_mask] = 1e6 * torch.abs(batch_J[invalid_mask])
        
        return batch_W
    

    def _compute_PK1_batch(self, batch_F):
        """
        Compute the First Piola-Kirchhoff stress tensor for a batch of deformation gradients
        
        Args:
            batch_F: Tensor of shape [batch_size, 3, 3] containing deformation gradients
            
        Returns:
            Tensor of shape [batch_size, 3, 3] containing PK1 tensors
        """
        # Compute J = det(F)
        batch_J = torch.linalg.det(batch_F)
        
        # Compute inverse of F (batched)
        batch_inv_F = torch.linalg.inv(batch_F)
        
        # Compute F^-T
        batch_inv_F_T = batch_inv_F.transpose(1, 2)
        
        # First Piola-Kirchhoff stress tensor (Pablo's formulation)
        # P = μ(F - F^-T) + 0.5λ(J²-1)F^-T
        term1 = self.mu * batch_F
        term2 = -self.mu * batch_inv_F_T
        term3 = 0.5 * self.lmbda * ((batch_J * batch_J - 1.0).unsqueeze(-1).unsqueeze(-1)) * batch_inv_F_T
        
        batch_P = term1 + term2 + term3
        
        return batch_P
    
    def forward(self, u_tensor):
        """
        Compute total elastic energy for the displacement field using optimized batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Total strain energy
        """
        # Ensure tensor is on the correct device and type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize total energy
        total_energy = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # Get number of quadrature points
        num_qp = len(self.quadrature_points)
        
        # Process elements in batches for memory efficiency
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Gather element coordinates and displacements for the entire batch
            batch_coords = self.coordinates[batch_elements]  # [batch, nodes_per_elem, 3]
            batch_disps = u[batch_elements]                 # [batch, nodes_per_elem, 3]
            
            # Initialize batch energy
            batch_energy = torch.zeros(batch_size, device=self.device, dtype=self.dtype)
            
            # Process all quadrature points
            for q_idx in range(num_qp):
                if self.precomputed:
                    # Use precomputed derivatives
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]  # [batch, nodes, 3]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]    # [batch]
                else:
                    # Compute derivatives on-the-fly (less efficient)
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Compute energy densities for all elements at once using Pablo's formulation
                batch_energy_density = self._compute_neohook_energy_density_batch(batch_F)
                
                # Numerical integration
                batch_energy += batch_energy_density * batch_detJ * self.quadrature_weights[q_idx]
            
            # Sum up all element energies in this batch
            total_energy += torch.sum(batch_energy)
            
            # Clean up memory periodically
            if torch.cuda.is_available() and batch_size > 500:
                torch.cuda.empty_cache()
        
        return total_energy
    
    def compute_batch_energy(self, batch_u):
        """
        Process batch of displacement fields efficiently
        
        Args:
            batch_u: Batch of displacement fields [batch_size, num_nodes * 3]
            
        Returns:
            Energies for each displacement field [batch_size]
        """
        batch_size = batch_u.shape[0]
        energies = torch.zeros(batch_size, dtype=self.dtype, device=self.device)
        
        # For larger batches, process in smaller chunks to save memory
        max_samples_per_batch = 4
        for i in range(0, batch_size, max_samples_per_batch):
            end_idx = min(i + max_samples_per_batch, batch_size)
            sub_batch = batch_u[i:end_idx]
            sub_batch_size = end_idx - i
            
            # Process each sample in sub-batch
            for j in range(sub_batch_size):
                energies[i + j] = self.forward(sub_batch[j])
            
            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return energies
    
    def compute_strains_stresses(self, u_tensor):
        """
        Compute strains and stresses across the domain using batch operations
        
        Args:
            u_tensor: Displacement field [num_nodes * 3]
            
        Returns:
            Dictionary with 'strains' and 'stresses' tensors
        """
        # Ensure tensor is on correct device/type
        if u_tensor.device != self.device or u_tensor.dtype != self.dtype:
            u_tensor = u_tensor.to(device=self.device, dtype=self.dtype)
        
        # Reshape displacement vector
        u = u_tensor.reshape(self.num_nodes, self.dim)
        
        # Initialize storage for results
        strains = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                             dtype=self.dtype, device=self.device)
        stresses = torch.zeros((self.num_elements, len(self.quadrature_points), 6), 
                              dtype=self.dtype, device=self.device)
        
        # Process elements in batches
        for batch_start in range(0, self.num_elements, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_elements)
            batch_elements = self.elements[batch_start:batch_end]
            batch_size = batch_end - batch_start
            
            # Get coordinates and displacements for this batch
            batch_coords = self.coordinates[batch_elements]
            batch_disps = u[batch_elements]
            
            # Process each quadrature point
            for q_idx in range(len(self.quadrature_points)):
                if self.precomputed:
                    batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx]
                    batch_detJ = self.detJ_all[batch_start:batch_end, q_idx]
                else:
                    # Compute on-the-fly
                    qp = self.quadrature_points[q_idx]
                    batch_dN_dx = []
                    batch_detJ = []
                    for i in range(batch_size):
                        dN_dx, detJ = self._compute_derivatives(batch_coords[i], qp)
                        batch_dN_dx.append(dN_dx)
                        batch_detJ.append(detJ)
                    batch_dN_dx = torch.stack(batch_dN_dx)
                    batch_detJ = torch.tensor(batch_detJ, device=self.device, dtype=self.dtype)
                
                # Compute deformation gradients for all elements at once
                batch_F = self._compute_F_batch(batch_coords, batch_disps, batch_dN_dx)
                
                # Batch identity matrix
                batch_I = torch.eye(3, dtype=self.dtype, device=self.device).expand(batch_size, 3, 3)
                
                # Batch Right Cauchy-Green tensor C = F^T F
                batch_C = torch.bmm(batch_F.transpose(1, 2), batch_F)
                
                # Batch Green-Lagrange strain tensor E = 0.5(C - I)
                batch_E = 0.5 * (batch_C - batch_I)
                
                # Calculate batch PK1 stress tensor using Pablo's formulation
                batch_P = self._compute_PK1_batch(batch_F)
                
                # Calculate batch Second Piola-Kirchhoff stress tensor S = F^{-1} P
                batch_inv_F = torch.linalg.inv(batch_F)
                batch_S = torch.bmm(batch_inv_F, batch_P)
                
                # Convert tensors to Voigt notation [xx, yy, zz, xy, yz, xz]
                for b in range(batch_size):
                    e_idx = batch_start + b
                    E = batch_E[b]
                    S = batch_S[b]
                    
                    strains[e_idx, q_idx] = torch.tensor([
                        E[0,0], E[1,1], E[2,2], E[0,1], E[1,2], E[0,2]
                    ], device=self.device, dtype=self.dtype)
                    
                    stresses[e_idx, q_idx] = torch.tensor([
                        S[0,0], S[1,1], S[2,2], S[0,1], S[1,2], S[0,2]
                    ], device=self.device, dtype=self.dtype)
        
        return {'strains': strains, 'stresses': stresses}
    
    def compute_PK1(self, F):
        """
        Compute the First Piola-Kirchhoff stress tensor from deformation gradient
        
        Args:
            F: Deformation gradient tensor [3, 3]
            
        Returns:
            First Piola-Kirchhoff stress tensor [3, 3]
        """
        # Compute J = det(F)
        J = torch.det(F)
        
        # Compute inverse of F
        inv_F = torch.linalg.inv(F)
        
        # First Piola-Kirchhoff stress tensor (Pablo's formulation)
        # P = μ(F - F^-T) + 0.5λ(J²-1)F^-T
        P = self.mu * (F - torch.transpose(inv_F, 0, 1)) + \
            0.5 * self.lmbda * (J * J - 1) * torch.transpose(inv_F, 0, 1)
        
        return P
    
    def compute_div_p(self, u_batch):
        """
        Fully vectorized implementation to compute the raw divergence of Piola-Kirchoff tensor.
        Returns the raw div(P) values without any transformation for post-processing in the training loop.
        
        Args:
            u_batch: Batch of displacement fields [batch_size, num_nodes * 3]
            
        Returns:
            Raw div(P) values [batch_size, num_nodes, 3]
        """
        batch_size = u_batch.shape[0]
        
        # Process the entire batch at once if memory permits, otherwise use chunking
        max_chunk_size = 4  # Adjust based on GPU memory
        div_p_all = []
        
        for chunk_start in range(0, batch_size, max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, batch_size)
            chunk_u_batch = u_batch[chunk_start:chunk_end]
            chunk_size = chunk_end - chunk_start
            
            # Reshape all displacements in the chunk
            u_reshaped = chunk_u_batch.reshape(chunk_size, self.num_nodes, self.dim)
            
            # Initialize divergence storage for all samples in chunk
            div_p = torch.zeros((chunk_size, self.num_nodes, 3), 
                            dtype=self.dtype, device=self.device)
            node_counts = torch.zeros((chunk_size, self.num_nodes), 
                                    dtype=self.dtype, device=self.device)
            
            # Process elements in batches for memory efficiency
            for batch_start in range(0, self.num_elements, self.batch_size):
                batch_end = min(batch_start + self.batch_size, self.num_elements)
                batch_elements = self.elements[batch_start:batch_end]
                local_batch_size = batch_end - batch_start
                
                # Get element coordinates and displacements
                batch_coords = self.coordinates[batch_elements].unsqueeze(0).expand(
                    chunk_size, local_batch_size, self.nodes_per_element, 3)
                    
                # Gather displacements for all elements in batch for all samples in chunk
                batch_disps = torch.stack([u_reshaped[b, batch_elements] for b in range(chunk_size)])
                
                # Initialize divergence contribution
                batch_div_p = torch.zeros((chunk_size, local_batch_size, self.nodes_per_element, 3), 
                                        dtype=self.dtype, device=self.device)
                
                # Compute for all quadrature points
                for q_idx in range(len(self.quadrature_points)):
                    if self.precomputed:
                        # Use precomputed derivatives
                        batch_dN_dx = self.dN_dx_all[batch_start:batch_end, q_idx].unsqueeze(0).expand(
                            chunk_size, -1, -1, -1)
                        batch_detJ = self.detJ_all[batch_start:batch_end, q_idx].unsqueeze(0).expand(
                            chunk_size, -1)
                    else:
                        # Compute derivatives vectorized
                        batch_dN_dx, batch_detJ = self._compute_derivatives_batch(batch_coords, 
                                                self.quadrature_points[q_idx])
                    
                    # Compute F tensors for all elements in batch for all samples in chunk
                    batch_F = self._compute_F_batch_multi(batch_coords, batch_disps, batch_dN_dx)
                    
                    # Compute PK1 stress for all elements
                    batch_P = self._compute_PK1_batch_multi(batch_F)
                    
                    # Vectorized divergence computation using einsum
                    qp_div_p = torch.einsum('cbij,cbnj->cbni', batch_P, batch_dN_dx)

                    # Weight by quadrature weight and jacobian determinant (no tanh)
                    weighted_div_p = qp_div_p * self.quadrature_weights[q_idx] * batch_detJ.unsqueeze(-1).unsqueeze(-1)
                    
                    # Accumulate divergence contribution
                    batch_div_p += weighted_div_p
                
                # Assemble element contributions to nodes
                for e in range(local_batch_size):
                    element_nodes = batch_elements[e].long()
                    div_p[:, element_nodes] += batch_div_p[:, e]
                    node_counts[:, element_nodes] += 1
            
            # Average contributions at nodes with multiple element memberships
            mask = node_counts > 0
            div_p[mask] = div_p[mask] / node_counts[mask].unsqueeze(-1)
            
            # Save this chunk's results
            div_p_all.append(div_p)
        
        # Combine all chunks
        return torch.cat(div_p_all, dim=0)
    

    # Add these methods to PabloNeoHookeanEnergy class

    def compute_volume_comparison(self, u_linear, u_total):
        """
        Compare volumes between original mesh, linear modes prediction,
        and neural network prediction.
        
        Args:
            u_linear: Linear displacement field [num_nodes * 3] or [batch_size, num_nodes * 3]
            u_total: Total displacement field [num_nodes * 3] or [batch_size, num_nodes * 3]
            
        Returns:
            Dictionary with volume information
        """
        # Handle batch dimension if present
        if len(u_linear.shape) > 1 and u_linear.shape[0] == 1:
            u_linear = u_linear.squeeze(0)
        if len(u_total.shape) > 1 and u_total.shape[0] == 1:
            u_total = u_total.squeeze(0)
        # Calculate volumes
        original_volume = self.compute_mesh_volume()
        linear_volume = self.compute_deformed_volume(u_linear)
        neural_volume = self.compute_deformed_volume(u_total)
        
        # Calculate volume ratios
        linear_volume_ratio = linear_volume / original_volume
        neural_volume_ratio = neural_volume / original_volume
        improvement_ratio = abs(neural_volume_ratio - 1.0) / abs(linear_volume_ratio - 1.0) if abs(linear_volume_ratio - 1.0) > 1e-10 else 1.0
        
        # Create result dictionary
        result = {
            'original_volume': original_volume,
            'linear_volume': linear_volume, 
            'neural_volume': neural_volume,
            'linear_volume_ratio': linear_volume_ratio,
            'neural_volume_ratio': neural_volume_ratio,
            'volume_preservation_improvement': improvement_ratio
        }
        
        return result

    def compute_mesh_volume(self):
        """Calculate the total volume of the original mesh"""
        if self.element_type == "tetrahedron":
            total_volume = 0.0
            for e in range(self.num_elements):
                element_nodes = self.elements[e].cpu().numpy()
                vertices = self.coordinates[element_nodes].cpu().numpy()
                
                # Calculate tetrahedron volume
                edges = vertices[1:] - vertices[0]
                volume = abs(np.linalg.det(edges)) / 6.0
                total_volume += volume
                
            return total_volume
        else:
            # For hexahedra, use a numerical approach with Gaussian quadrature
            total_volume = 0.0
            
            for e in range(self.num_elements):
                if self.precomputed:
                    # Use precomputed Jacobian determinants
                    element_volume = 0.0
                    for q_idx in range(len(self.quadrature_points)):
                        element_volume += self.detJ_all[e, q_idx].item() * self.quadrature_weights[q_idx].item()
                    total_volume += element_volume
                else:
                    element_nodes = self.elements[e].cpu().numpy()
                    vertices = self.coordinates[element_nodes].cpu().numpy()
                    
                    element_volume = 0.0
                    for q_idx in range(len(self.quadrature_points)):
                        tensor_vertices = torch.tensor(vertices, device=self.device, dtype=self.dtype)
                        _, detJ = self._compute_derivatives(tensor_vertices, self.quadrature_points[q_idx])
                        element_volume += detJ.item() * self.quadrature_weights[q_idx].item()
                        
                    total_volume += element_volume
                    
            return total_volume

    def compute_deformed_volume(self, displacement):
        """
        Calculate the total volume of the mesh after applying a displacement field
        
        Args:
            displacement: Displacement field [num_nodes * 3] or [batch_size, num_nodes * 3]
            
        Returns:
            Total volume of the deformed mesh
        """
        # Ensure displacement is in correct format
        if isinstance(displacement, torch.Tensor):
            displacement = displacement.detach().cpu().numpy()
        
        # Reshape displacement correctly
        if len(displacement.shape) == 1:
            # Single vector: reshape to [num_nodes, 3]
            displacement = displacement.reshape(-1, 3)
        elif len(displacement.shape) == 2 and displacement.shape[0] == 1:
            # Batch with single sample: reshape to [num_nodes, 3]
            displacement = displacement.reshape(-1, 3)
        
        # Verify shapes match
        coordinates = self.coordinates.cpu().numpy()
        if displacement.shape[0] != coordinates.shape[0]:
            # Try reshaping if total elements match
            if displacement.size == coordinates.size:
                displacement = displacement.reshape(coordinates.shape)
            else:
                raise ValueError(f"Displacement shape {displacement.shape} cannot be reshaped to match coordinates {coordinates.shape}")
        
        # Get deformed coordinates
        deformed_coords = coordinates + displacement
        
        # Rest of function remains the same...
        if self.element_type == "tetrahedron":
            total_volume = 0.0
            for e in range(self.num_elements):
                element_nodes = self.elements[e].cpu().numpy()
                vertices = deformed_coords[element_nodes]
                
                # Calculate tetrahedron volume
                edges = vertices[1:] - vertices[0]
                volume = abs(np.linalg.det(edges)) / 6.0
                total_volume += volume
                
            return total_volume
        else:
            # For hexahedra
            total_volume = 0.0
            for e in range(self.num_elements):
                element_nodes = self.elements[e].cpu().numpy()
                vertices = deformed_coords[element_nodes]
                
                element_volume = 0.0
                for q_idx in range(len(self.quadrature_points)):
                    tensor_vertices = torch.tensor(vertices, device=self.device, dtype=self.dtype)
                    _, detJ = self._compute_derivatives(tensor_vertices, self.quadrature_points[q_idx])
                    element_volume += detJ.item() * self.quadrature_weights[q_idx].item()
                    
                total_volume += element_volume
                
            return total_volume

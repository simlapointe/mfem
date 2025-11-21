// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;

#ifdef MFEM_USE_MPI

TEST_CASE("Refine Rebalance", "[Parallel]")
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const int order = 1; // ND p>1 not currently supported
   const int dim = 3;
   const double tol = 1.0e-12;

   // Create simple tet mesh
   Mesh serial_mesh = Mesh::MakeCartesian3D(5, 5, 5, Element::TETRAHEDRON);
   serial_mesh.EnsureNCMesh(true);

   // Create simple functions that can be represented exactly in the FE space
   VectorFunctionCoefficient vcoef(dim, [](const Vector &x, Vector &v) {
      v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
   });

   FunctionCoefficient scoef([](const Vector &x) {
      return x(0) + 2.0*x(1) + 3.0*x(2);
   });

   // Loop over FE collections
   std::vector<std::pair<std::string, FiniteElementCollection*>> fecs = {
      {"ND", new ND_FECollection(order, dim)},
      {"H1", new H1_FECollection(order, dim)},
      //{"RT", new RT_FECollection(order, dim)}, // RT not currently supported
      {"L2", new L2_FECollection(order, dim)}
   };

   for (auto& [name, fec] : fecs)
   {
      // Original mesh (create three copies)
      ParMesh mesh0(MPI_COMM_WORLD, serial_mesh); // Original
      ParMesh mesh1(MPI_COMM_WORLD, serial_mesh); // Will be refined
      ParMesh mesh2(MPI_COMM_WORLD, serial_mesh); // Will be refined and rebalanced

      // Refine every element on one rank only to create unbalanced mesh
      Array<int> elements;
      if (rank == 0) for (int i = 0; i < mesh1.GetNE(); i++) elements.Append(i);
      mesh1.GeneralRefinement(elements, -1, 1);
      mesh2.GeneralRefinement(elements, -1, 1);

      // Create finite element spaces for each mesh
      ParFiniteElementSpace fe0(&mesh0, fec);
      ParFiniteElementSpace fe1(&mesh1, fec);
      ParFiniteElementSpace fe2(&mesh2, fec);

      // Choose coefficient based on FE space type
      bool is_vector = (fec->GetRangeType(dim) == FiniteElement::VECTOR);

      // Initial grid function
      ParGridFunction gf0(&fe0);
      if (is_vector) gf0.ProjectCoefficient(vcoef);
      else gf0.ProjectCoefficient(scoef);
      gf0.ExchangeFaceNbrData();
      double err0 = is_vector ? gf0.ComputeMaxError(vcoef) : gf0.ComputeMaxError(scoef);
      REQUIRE(err0 < tol);

      // Test refinement operator
      auto refine_op_01 = std::make_unique<TransferOperator>(fe0, fe1);
      ParGridFunction gf1(&fe1);
      refine_op_01->Mult(gf0, gf1);
      gf1.ExchangeFaceNbrData();
      double err1 = is_vector ? gf1.ComputeMaxError(vcoef) : gf1.ComputeMaxError(scoef);
      REQUIRE(err1 < tol);

      // Test rebalance operator
      mesh2.Rebalance();
      auto rebalance_op_12 = std::unique_ptr<Operator>(const_cast<Operator*>(fe2.GetUpdateOperator()));
      fe2.SetUpdateOperatorOwner(false);
      ParGridFunction gf2(&fe2);
      rebalance_op_12->Mult(gf1, gf2);
      gf2.ExchangeFaceNbrData();
      double err2 = is_vector ? gf2.ComputeMaxError(vcoef) : gf2.ComputeMaxError(scoef);
      REQUIRE(err2 < tol);

      // Project coefficients directly on the rebalanced fe2
      ParGridFunction gf2_fresh(&fe2);
      if (is_vector) gf2_fresh.ProjectCoefficient(vcoef);
      else gf2_fresh.ProjectCoefficient(scoef);
      double err2_fresh = is_vector ? gf2_fresh.ComputeMaxError(vcoef) : gf2_fresh.ComputeMaxError(scoef);
      REQUIRE(err2_fresh < tol);

      // Compare norm of (gf2 - gf2_fresh)
      gf2_fresh -= gf2;
      REQUIRE(gf2_fresh.Norml2() < tol);

      // Copy rebalanced mesh
      ParMesh mesh3(mesh2); // Refined, rebalanced, and refined again
      ParMesh mesh4(mesh2); // Refined, rebalanced, refined again, and rebalanced again

      // Refine the rebalanced meshes non-uniformly
      Array<int> elements2;
      if (rank == 1) for (int i = 0; i < mesh3.GetNE(); i++) elements2.Append(i);
      mesh3.GeneralRefinement(elements2, -1, 1);
      mesh4.GeneralRefinement(elements2, -1, 1);

      // Create finite element spaces
      ParFiniteElementSpace fe3(&mesh3, fec);
      ParFiniteElementSpace fe4(&mesh4, fec);

      // Test refinement operator
      auto refine_op_23 = std::make_unique<TransferOperator>(fe2, fe3);
      ParGridFunction gf3(&fe3);
      refine_op_23->Mult(gf2, gf3);
      gf3.ExchangeFaceNbrData();
      double err3 = is_vector ? gf3.ComputeMaxError(vcoef) : gf3.ComputeMaxError(scoef);
      REQUIRE(err3 < tol);

      // Test rebalance operator
      mesh4.Rebalance();
      auto rebalance_op_34 = std::unique_ptr<Operator>(const_cast<Operator*>(fe4.GetUpdateOperator()));
      fe4.SetUpdateOperatorOwner(false);
      ParGridFunction gf4(&fe4);
      rebalance_op_34->Mult(gf3, gf4);
      gf4.ExchangeFaceNbrData();
      double err4 = is_vector ? gf4.ComputeMaxError(vcoef) : gf4.ComputeMaxError(scoef);
      REQUIRE(err4 < tol);

      // Project coefficients directly on the rebalanced fe4
      ParGridFunction gf4_fresh(&fe4);
      if (is_vector) gf4_fresh.ProjectCoefficient(vcoef);
      else gf4_fresh.ProjectCoefficient(scoef);
      double err4_fresh = is_vector ? gf4_fresh.ComputeMaxError(vcoef) : gf4_fresh.ComputeMaxError(scoef);
      REQUIRE(err4_fresh < tol);

      // Compare norm of (gf4 - gf4_fresh)
      gf4_fresh -= gf4;
      REQUIRE(gf4_fresh.Norml2() < tol);
   }

   // Cleanup
   for (auto& [name, fec] : fecs) delete fec;
}

#endif // #ifdef MFEM_USE_MPI
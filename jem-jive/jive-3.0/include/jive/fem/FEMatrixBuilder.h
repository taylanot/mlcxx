
/*
 *  Copyright (C) 2019 DRG. All rights reserved.
 *
 *  This file is part of Jive, an object oriented toolkit for solving
 *  partial differential equations.
 *
 *  Commercial License Usage
 *
 *  This file may be used under the terms of a commercial license
 *  provided with the software, or under the terms contained in a written
 *  agreement between you and DRG. For more information contact DRG at
 *  http://www.dynaflow.com.
 *
 *  GNU Lesser General Public License Usage
 *
 *  Alternatively, this file may be used under the terms of the GNU
 *  Lesser General Public License version 2.1 or version 3 as published
 *  by the Free Software Foundation and appearing in the file
 *  LICENSE.LGPLv21 and LICENSE.LGPLv3 included in the packaging of this
 *  file. Please review the following information to ensure the GNU
 *  Lesser General Public License requirements will be met:
 *  https://www.gnu.org/licenses/lgpl.html and
 *  http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
 *  This file is part of Jive, an object oriented toolkit for
 *  solving partial differential equations.
 *
 *  Jive version: 3.0
 *  Date:         Fri 20 Dec 14:30:12 CET 2019
 */

#ifndef JIVE_FEM_FEMATRIXBUILDER_H
#define JIVE_FEM_FEMATRIXBUILDER_H

#include <jive/algebra/SparseMatrixBuilder.h>
#include <jive/fem/import.h>
#include <jive/fem/ElementSet.h>


JIVE_BEGIN_PACKAGE( fem )


//-----------------------------------------------------------------------
//   class FEMatrixBuilder
//-----------------------------------------------------------------------


class FEMatrixBuilder : public algebra::SparseMatrixBuilder
{
 public:

  JEM_DECLARE_CLASS       ( FEMatrixBuilder,
                            algebra::SparseMatrixBuilder );

  typedef algebra::
    MatrixBuilder           MBuilder;

  static const char*        TYPE_NAME;


                            FEMatrixBuilder

    ( const String&           name,
      const ElementSet&       elems,
      const Ref<DofSpace>&    dofs,
      Ref<SparseMatrixObj>    matrix = nullptr );

  virtual void              clear             () override;
  virtual void              setToZero         () override;
  virtual void              updateStructure   ();

  virtual void              setStructure

    ( const SparseStruct&     st )               override;

  static Ref<MBuilder>      makeNew

    ( const String&           name,
      const Properties&       conf,
      const Properties&       props,
      const Properties&       params,
      const Properties&       globdat );

  static void               declare           ();


 protected:

  virtual                  ~FEMatrixBuilder   ();


 private:

  void                      connect_          ();
  void                      invalidate_       ();
  void                      updateStructure_  ();


 private:

  ElementSet                elems_;
  Ref<DofSpace>             dofs_;

  bool                      updated_;

};


JIVE_END_PACKAGE( fem )

#endif

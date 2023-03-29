
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

#ifndef JIVE_UTIL_PRIVATE_CONSTRAINTSIMP_H
#define JIVE_UTIL_PRIVATE_CONSTRAINTSIMP_H

#include <jem/base/Object.h>
#include <jive/Array.h>
#include <jive/SparseMatrix.h>
#include <jive/util/import.h>
#include <jive/util/EventMask.h>


JIVE_BEGIN_PACKAGE( util )


class DofSpace;
class Reordering;
class Constraints;


//-----------------------------------------------------------------------
//   class ConstraintsImp
//-----------------------------------------------------------------------


class ConstraintsImp : public Object
{
 public:

  typedef Object            Super;
  typedef ConstraintsImp    Self;

  friend class              Constraints;

  static const char*        ERR_CONTEXT;
  static const int          NEW_RVALS       = 1 << 0;
  static const int          NEW_STRUCT      = 1 << 1;


  virtual Ref<Self>         clone           () const = 0;
  virtual void              clear           ()       = 0;

  virtual void              reserve

    ( idx_t                   slaveCount );

  virtual void              trimToSize      ();
  virtual idx_t             slaveDofCount   () const = 0;
  virtual idx_t             masterDofCount  () const = 0;

  virtual idx_t             masterDofCount

    ( idx_t                   idof )           const = 0;

  virtual bool              isSlaveDof

    ( idx_t                   idof )           const = 0;

  virtual bool              isMasterDof

    ( idx_t                   jdof )           const = 0;

  virtual bool              hasRvalue

    ( idx_t                   idof )           const = 0;

  virtual idx_t             getSlaveDofs

    ( const IdxVector&        idofs )          const = 0;

  virtual idx_t             getMasterDofs

    ( const IdxVector&        jdofs,
      idx_t                   idofs )          const = 0;

  virtual idx_t             getConstraint

    ( double&                 rval,
      const IdxVector&        jdofs,
      const Vector&           coeffs,
      idx_t                   idof )           const = 0;

  virtual void              setRvalue

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask )                = 0;

  virtual void              setRvalues

    ( const IdxVector&        idofs,
      const Vector&           rvals  )               = 0;

  virtual void              unsetRvalues

    ( const IdxVector&        idofs )                = 0;

  virtual double            getRvalue

    ( idx_t                   idof )           const = 0;

  virtual void              getRvalues

    ( const Vector&           rvals,
      const IdxVector&        idofs )          const = 0;

  virtual void              getRvalues

    ( const SparseMatrix&     cmat )           const = 0;

  virtual BoolVector        getRvalueMask   () const = 0;

  virtual void              getRvalueMask

    ( const BoolVector&       rmask,
      const IdxVector&        idofs )          const = 0;

  virtual void              setSlaveDofs

    ( const Vector&           vec,
      double                  rscale )         const = 0;

  virtual void              evalSlaveDofs

    ( const Vector&           vec,
      double                  rscale )         const = 0;

  virtual void              evalMasterDofs

    ( const Vector&           vec )            const = 0;

  virtual SparseMatrix      toMatrix        () const = 0;

  virtual void              printTo

    ( PrintWriter&            out )            const = 0;

  virtual void              newDofCount     ()       = 0;

  virtual void              newDofOrder

    ( const Reordering&       reord )                = 0;

  inline void               checkSize

    ( const Vector&           vec )            const;

  inline void               checkDofIndex

    ( idx_t                   idof );

  inline bool               checkDofIndex

    ( idx_t                   idof )           const;

  void                      checkDofIndices

    ( const IdxVector&        idofs );

  void                      sizeError

    ( const Vector&           vec )            const;

  void                      resolveError

    ( idx_t                   idof )           const;

  void                      dofIndexError

    ( idx_t                   idof )           const;


 public:

  EventMask                 events;


 protected:

  explicit                  ConstraintsImp

    ( const Ref<DofSpace>&    dofs);

                            ConstraintsImp

    ( const Self&             rhs );

  virtual                  ~ConstraintsImp  ();


 protected:

  Ref<DofSpace>             dofs_;
  idx_t                     dofCount_;


 private:

  void                      checkDofIndex_

    ( idx_t                   idof );

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   checkSize
//-----------------------------------------------------------------------


inline void ConstraintsImp::checkSize ( const Vector& vec ) const
{
  if ( vec.size() < dofCount_ )
  {
    sizeError ( vec );
  }
}


//-----------------------------------------------------------------------
//   checkDofIndex
//-----------------------------------------------------------------------


inline void ConstraintsImp::checkDofIndex ( idx_t idof )
{
  if ( idof < 0 || idof >= dofCount_ )
  {
    checkDofIndex_ ( idof );
  }
}


inline bool ConstraintsImp::checkDofIndex ( idx_t idof ) const
{
  if ( idof < 0 )
  {
    dofIndexError ( idof );
  }

  return (idof < dofCount_);
}


JIVE_END_PACKAGE( util )

#endif

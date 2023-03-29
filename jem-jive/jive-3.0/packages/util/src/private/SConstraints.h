
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

#ifndef JIVE_UTIL_PRIVATE_SCONSTRAINTS_H
#define JIVE_UTIL_PRIVATE_SCONSTRAINTS_H

#include <jem/base/utilities.h>
#include <jem/util/Flex.h>
#include <jem/util/FastAllocator.h>
#include <jive/util/private/ConstraintsImp.h>


JIVE_BEGIN_PACKAGE( util )


using jem::util::Flex;
using jem::util::FastAllocator;


//-----------------------------------------------------------------------
//   class SConstraints
//-----------------------------------------------------------------------


class SConstraints : public ConstraintsImp
{
 public:

  typedef ConstraintsImp    Super;
  typedef SConstraints      Self;

  struct                    SlaveNode;
  struct                    MasterNode;

  struct                    SlaveNode
  {
    uidx_t                    idof;
    double                    rvalue;
    MasterNode*               master;
  };

  struct                    MasterNode
  {
    idx_t                     jdof;
    idx_t                     next;
    double                    coeff;
  };


  explicit                  SConstraints

    ( const Ref<DofSpace>&    dofs );

                            SConstraints

    ( const Ref<DofSpace>&    dofs,
      const SparseMatrix&     cmat,
      const BoolVector&       rmask );

                            SConstraints

    ( const Self&             rhs );

  virtual Ref<Super>        clone           () const override;
  virtual void              clear           ()       override;

  virtual void              reserve

    ( idx_t                   slaveCount )           override;

  virtual void              trimToSize      ()       override;
  virtual idx_t             slaveDofCount   () const override;
  virtual idx_t             masterDofCount  () const override;

  virtual idx_t             masterDofCount

    ( idx_t                   idof )           const override;

  virtual bool              isSlaveDof

    ( idx_t                   idof )           const override;

  virtual bool              isMasterDof

    ( idx_t                   jdof )           const override;

  virtual bool              hasRvalue

    ( idx_t                   idof )           const override;

  virtual idx_t             getSlaveDofs

    ( const IdxVector&        idofs )          const override;

  virtual idx_t             getMasterDofs

    ( const IdxVector&        jdofs,
      idx_t                   idofs )          const override;

  void                      addConstraint

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask );

  void                      addConstraint

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask,
      idx_t                   jdof,
      double                  coeff );

  bool                      eraseConstraint

    ( idx_t                   idof );

  idx_t                     eraseConstraints

    ( const IdxVector&        idofs );

  virtual idx_t             getConstraint

    ( double&                 rval,
      const IdxVector&        jdofs,
      const Vector&           coeffs,
      idx_t                   idof )           const override;

  virtual void              setRvalue

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask )                override;

  virtual void              setRvalues

    ( const IdxVector&        idofs,
      const Vector&           rvals  )               override;

  virtual void              unsetRvalues

    ( const IdxVector&        idofs )                override;

  virtual double            getRvalue

    ( idx_t                   idof )           const override;

  virtual void              getRvalues

    ( const Vector&           rvals,
      const IdxVector&        idofs )          const override;

  virtual void              getRvalues

    ( const SparseMatrix&     cmat )           const override;

  virtual BoolVector        getRvalueMask   () const override;

  virtual void              getRvalueMask

    ( const BoolVector&       rmask,
      const IdxVector&        idofs )          const override;

  virtual void              setSlaveDofs

    ( const Vector&           vec,
      double                  rscale )         const override;

  virtual void              evalSlaveDofs

    ( const Vector&           vec,
      double                  rscale )         const override;

  virtual void              evalMasterDofs

    ( const Vector&           vec )            const override;

  virtual SparseMatrix      toMatrix        () const override;

  virtual void              printTo

    ( PrintWriter&            out )            const override;

  virtual void              newDofCount     ()       override;

  virtual void              newDofOrder

    ( const Reordering&       reord )                override;


 protected:

  virtual                  ~SConstraints    ();


 private:

  void                      freeAll_        ();

  idx_t                     newSlave_

    ( idx_t                   idof,
      double                  rval,
      bool                    rmask );

  idx_t                     newSlave_

    ( const SlaveNode&        orig );

  void                      freeSlave_

    ( idx_t                   islave );

  inline void               newMaster_

    ( SlaveNode&              slave,
      idx_t                   jdof,
      double                  coeff );

  inline void               freeMaster_

    ( SlaveNode&              slave );

  void                      newDofCount_    ();


 private:

  FastAllocator             allocator_;
  Flex<SlaveNode>           slaves_;
  Flex<idx_t>               dofConMap_;

  idx_t                     masterCount_;

};


JIVE_END_PACKAGE( util )

#endif

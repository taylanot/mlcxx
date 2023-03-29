
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

#ifndef JIVE_SOLVER_SCHURDOFSPACE_H
#define JIVE_SOLVER_SCHURDOFSPACE_H

#include <jive/util/DofSpace.h>
#include <jive/mp/BorderSet.h>
#include <jive/solver/import.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class SchurDofSpace
//-----------------------------------------------------------------------


class SchurDofSpace : public DofSpace
{
 public:

  JEM_DECLARE_CLASS       ( SchurDofSpace, DofSpace );


                            SchurDofSpace

    ( const Ref<DofSpace>&    dofs,
      const BorderSet&        borders );

                            SchurDofSpace

    ( const Ref<DofSpace>&    dofs,
      const BorderSet&        recvBorders,
      const BorderSet&        sendBorders );

  void                      start             ();
  void                      finish            ();
  void                      update            ();

  virtual idx_t             typeCount         () const override;
  virtual idx_t             dofCount          () const override;
  virtual void              resetEvents       ()       override;

  virtual idx_t             findType

    ( const String&           name )             const override;

  virtual String            getTypeName

    ( idx_t                   itype )            const override;

  virtual StringVector      getTypeNames      () const override;

  virtual idx_t             findDofIndex

    ( idx_t                   iitem,
      idx_t                   itype )            const override;

  virtual idx_t             findDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )           const override;

  virtual idx_t             collectDofIndices

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      const IdxVector&        itypes )           const override;

  virtual idx_t             getDofsForItem

    ( const IdxVector&        idofs,
      const IdxVector&        itypes,
      idx_t                   iitem )            const override;

  virtual idx_t             getDofsForType

    ( const IdxVector&        idofs,
      const IdxVector&        iitems,
      idx_t                   itype )            const override;

  virtual void              decodeDofIndex

    ( idx_t&                  iitem,
      idx_t&                  itype,
      idx_t                   idof   )           const override;

  virtual void              decodeDofIndices

    ( const IdxVector&        iitems,
      const IdxVector&        itypes,
      const IdxVector&        idofs  )           const override;

  virtual SparseIdxMatrix   toMatrix          () const override;
  virtual ItemSet*          getItems          () const override;
  inline IdxVector          getDofMap         () const;
  inline IdxVector          getInnerDofs      () const;
  inline IdxVector          getBorderDofs     () const;
  inline DofSpace*          getTotalDofSpace  () const;
  inline BorderSet          getRecvBorders    () const;
  inline BorderSet          getSendBorders    () const;


 protected:

  virtual                  ~SchurDofSpace     ();

  virtual void              emitEvents_

    ( int                     events )                 override;


 private:

  void                      init_             ();
  void                      update_           ();
  void                      invalidate_       ();


 private:

  Ref<DofSpace>             dofs_;
  BorderSet                 recvBorders_;
  BorderSet                 sendBorders_;

  IdxVector                 dofMap_;
  IdxVector                 innerDofs_;
  IdxVector                 borderDofs_;

  bool                      updated_;
  idx_t                     started_;

};




//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getDofMap
//-----------------------------------------------------------------------


inline IdxVector SchurDofSpace::getDofMap () const
{
  return dofMap_;
}


//-----------------------------------------------------------------------
//   getInnerDofs
//-----------------------------------------------------------------------


inline IdxVector SchurDofSpace::getInnerDofs () const
{
  return innerDofs_;
}


//-----------------------------------------------------------------------
//   getBorderDofs
//-----------------------------------------------------------------------


inline IdxVector SchurDofSpace::getBorderDofs () const
{
  return borderDofs_;
}


//-----------------------------------------------------------------------
//   getTotalDofSpace
//-----------------------------------------------------------------------


inline DofSpace* SchurDofSpace::getTotalDofSpace () const
{
  return dofs_.get ();
}


//-----------------------------------------------------------------------
//   getRecvBorders
//-----------------------------------------------------------------------


inline BorderSet SchurDofSpace::getRecvBorders () const
{
  return recvBorders_;
}


//-----------------------------------------------------------------------
//   getSendBorders
//-----------------------------------------------------------------------


inline BorderSet SchurDofSpace::getSendBorders () const
{
  return sendBorders_;
}


JIVE_END_PACKAGE( solver )

#endif

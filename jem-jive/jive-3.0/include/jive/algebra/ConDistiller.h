
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

#ifndef JIVE_ALGEBRA_CONDISTILLER_H
#define JIVE_ALGEBRA_CONDISTILLER_H

#include <jem/base/Object.h>
#include <jem/io/Serializable.h>
#include <jive/Array.h>
#include <jive/algebra/import.h>


JIVE_BEGIN_PACKAGE( algebra )


//-----------------------------------------------------------------------
//   class ConDistiller
//-----------------------------------------------------------------------


class ConDistiller : public Object, public Serializable
{
 public:

  JEM_DECLARE_CLASS     ( ConDistiller, Object );


                          ConDistiller      ();

  explicit                ConDistiller

    ( Ref<Constraints>      cons,
      Ref<XDofSpace>        rdofs = nullptr );

  virtual void            readFrom

    ( ObjectInput&          in )                     override;

  virtual void            writeTo

    ( ObjectOutput&         out )              const override;

  inline void             update            ();
  inline bool             updated           () const;

  void                    reduce

    ( const Vector&         r,
      const Vector&         f )                const;

  void                    rdaxpy

    ( const Vector&         r,
      double                s,
      const Vector&         f )                const;

  void                    filter

    ( const Vector&         r,
      const Vector&         f )                const;

  void                    expand

    ( const Vector&         f,
      const Vector&         r )                const;

  IdxVector               getDofMap         () const;
  DofSpace*               getDofSpace       () const;
  DofSpace*               getReducedDofs    () const;
  Constraints*            getConstraints    () const;
  Constraints*            getReducedCons    () const;

  inline const idx_t*     getOffsetPtr      () const;
  inline const idx_t*     getIndexPtr       () const;
  inline const double*    getCoeffPtr       () const;


 protected:

  virtual                ~ConDistiller      ();


 private:

  void                    connect_          ();
  void                    update_           ();
  void                    updateStruct_     ();
  void                    updateRvalues_    ();
  void                    structChanged_    ();
  void                    rvaluesChanged_   ();

  void                    rdofsResized_

    ( idx_t                 size );

  void                    rdofsReordered_

    ( const Reordering&     reord );

  void                    printInfo_        ();
  static void             nanError_         ();


 private:

  static const int        NEW_STRUCT_;
  static const int        NEW_RVALUES_;

  Ref<Constraints>        cons_;
  Ref<Constraints>        rcons_;
  Ref<XDofSpace>          rdofs_;

  IdxVector               dofMap_;
  IdxVector               conOffsets_;
  IdxVector               dofIndices_;
  Vector                  conCoeffs_;

  idx_t                   dofCount_;
  idx_t                   rdofCount_;
  int                     dirtyBits_;

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   update
//-----------------------------------------------------------------------


inline void ConDistiller::update ()
{
  if ( dirtyBits_ )
  {
    update_ ();
  }
}


//-----------------------------------------------------------------------
//   updated
//-----------------------------------------------------------------------


inline bool ConDistiller::updated () const
{
  return (dirtyBits_ == 0);
}


//-----------------------------------------------------------------------
//   getOffsetPtr
//-----------------------------------------------------------------------


inline const idx_t* ConDistiller::getOffsetPtr () const
{
  return conOffsets_.addr ();
}


//-----------------------------------------------------------------------
//   getIndexPtr
//-----------------------------------------------------------------------


inline const idx_t* ConDistiller::getIndexPtr () const
{
  return dofIndices_.addr ();
}


//-----------------------------------------------------------------------
//   getCoeffPtr
//-----------------------------------------------------------------------


inline const double* ConDistiller::getCoeffPtr () const
{
  return conCoeffs_.addr ();
}


JIVE_END_PACKAGE( algebra )

#endif

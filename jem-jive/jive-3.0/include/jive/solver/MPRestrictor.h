
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

#ifndef JIVE_SOLVER_MPRESTRICTOR_H
#define JIVE_SOLVER_MPRESTRICTOR_H

#include <jive/algebra/MPMatrixExtension.h>
#include <jive/solver/Restrictor.h>


JIVE_BEGIN_PACKAGE( solver )


//-----------------------------------------------------------------------
//   class MPRestrictor
//-----------------------------------------------------------------------


class MPRestrictor : public Restrictor,
                     public MPMatrixExt
{
 public:

  JEM_DECLARE_CLASS       ( MPRestrictor, Restrictor );


                            MPRestrictor

    ( Ref<Restrictor>         rt,
      Ref<VectorExchanger>    vex );

  virtual void              start             ()       override;
  virtual void              finish            ()       override;
  virtual void              update            ()       override;
  bool                      updated           () const;

  virtual void              resetEvents       ()       override;
  virtual Shape             shape             () const override;

  virtual void              matmul

    ( const Vector&           lhs,
      const Vector&           rhs )              const override;

  virtual void              transmul

    ( const Vector&           lhs,
      const Vector&           rhs )              const override;

  void                      multiTransmul

    ( Matrix&                 lhsVecs,
      IdxVector&              lhsTags,
      const Matrix&           rhsVecs,
      const IdxVector&        rhsTags )          const;

  virtual void              configure

    ( const Properties&       props )                  override;

  virtual void              getConfig

    ( const Properties&       props )            const override;

  virtual bool              hasTrait

    ( const String&           trait )            const override;

  virtual void*             getExtByID

    ( ExtensionID             extID )            const override;

  virtual VectorExchanger*  getExchanger      () const override;
  virtual int               getExchangeMode   () const override;
  Vector                    getWeights        () const;
  inline Restrictor*        getInner          () const;


 protected:

  virtual                  ~MPRestrictor      ();


 private:

  void                      connect_          ();
  void                      update_           ();
  void                      valuesChanged_    ();
  void                      structChanged_    ();
  void                      syncEvents_       ();

  void                      setEvents_

    ( int                     events );


 private:

  static const int          NEW_VALUES_;
  static const int          NEW_STRUCT_;

  Ref<Restrictor>           inner_;
  Ref<VectorExchanger>      exchanger_;

  Vector                    vbuffer_;
  Vector                    weights_;

  int                       events_;
  idx_t                     started_;

};





//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   getInner
//-----------------------------------------------------------------------


inline Restrictor* MPRestrictor::getInner () const
{
  return inner_.get ();
}


JIVE_END_PACKAGE( solver )

#endif

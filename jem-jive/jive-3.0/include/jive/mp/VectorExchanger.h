
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

#ifndef JIVE_MP_VECTOREXCHANGER_H
#define JIVE_MP_VECTOREXCHANGER_H

#include <jive/Array.h>
#include <jive/mp/import.h>
#include <jive/mp/BorderSet.h>


JIVE_BEGIN_PACKAGE( mp )


//-----------------------------------------------------------------------
//   enum ExchangeMode
//-----------------------------------------------------------------------


enum ExchangeMode
{
  EXCHANGE,
  SCATTER
};


//-----------------------------------------------------------------------
//   class VectorExchanger
//-----------------------------------------------------------------------


class VectorExchanger : public Object
{
 public:

  JEM_DECLARE_CLASS       ( VectorExchanger, Object );


                            VectorExchanger

    ( const Ref<Context>&     mpx,
      const Ref<DofSpace>&    dofs,
      const BorderSet&        borders );

                            VectorExchanger

    ( const Ref<Context>&     mpx,
      const Ref<DofSpace>&    dofs,
      const BorderSet&        recvBorders,
      const BorderSet&        sendBorders );

  virtual String            toString          () const override;

  void                      barrier           ();

  void                      startOne

    ( int                     mode );

  inline void               endOne

    ( const Vector&           vec );

  void                      endOne

    ( const Vector&           recvVec,
      const Vector&           sendVec );

  void                      startSome

    ( int                     mode,
      const IdxVector&        sendTags );

  void                      endSome

    ( Matrix&                 recvVecs,
      IdxVector&              recvTags,
      const Matrix&           sendVecs,
      const IdxVector&        sendTags );

  inline void               exchange

    ( const Vector&           vec );

  inline void               exchange

    ( const Vector&           recvVec,
      const Vector&           sendVec );

  inline void               exchange

    ( Matrix&                 recvVecs,
      IdxVector&              recvTags,
      const Matrix&           sendVecs,
      const IdxVector&        sendTags );

  inline void               scatter

    ( const Vector&           vec );

  inline void               scatter

    ( const Vector&           recvVec,
      const Vector&           sendVec );

  inline void               scatter

    ( Matrix&                 recvVecs,
      IdxVector&              recvTags,
      const Matrix&           sendVecs,
      const IdxVector&        sendTags );

  inline Context*           getMPContext      () const;
  inline DofSpace*          getDofSpace       () const;
  inline BorderSet          getRecvBorders    () const;
  inline BorderSet          getSendBorders    () const;
  inline bool               hasOverlap        () const;
  bool                      isDistributed     () const;

  IdxVector                 getRecvDofs

    ( int                     mode )             const;

  IdxVector                 getSendDofs

    ( int                     mode )             const;

  void                      store

    ( const Properties&       globdat )          const;

  static Ref<Self>          find

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );

  static Ref<Self>          get

    ( const Ref<DofSpace>&    dofs,
      const Properties&       globdat );


 protected:

  virtual                  ~VectorExchanger   ();


 private:

  class                     XData_;


  void                      init_             ();
  void                      update_           ();
  void                      invalidate_       ();

  void                      xmodeError_

    ( const String&           where )            const;

  static void               invalidError_

    ( const String&           where );


 private:

  static const int          INACTIVE_;

  Ref<Context>              mpx_;
  Ref<DofSpace>             dofs_;

  BorderSet                 recvBorders_;
  BorderSet                 sendBorders_;

  idx_t                     dofCount_;

  int                       xmode_;
  bool                      overlap_;
  bool                      updated_;

  Ref<XData_>               xdata_[2];

};



//#######################################################################
//   Implementation
//#######################################################################

//-----------------------------------------------------------------------
//   endOne
//-----------------------------------------------------------------------


inline void VectorExchanger::endOne ( const Vector& vec )
{
  endOne ( vec, vec );
}


//-----------------------------------------------------------------------
//   exhange
//-----------------------------------------------------------------------


inline void VectorExchanger::exchange ( const Vector& vec )
{
  startOne ( EXCHANGE );
  endOne   ( vec, vec );
}


inline void VectorExchanger::exchange

  ( const Vector&  recvVec,
    const Vector&  sendVec )

{
  startOne ( EXCHANGE );
  endOne   ( recvVec, sendVec );
}


inline void VectorExchanger::exchange

  ( Matrix&           recvVecs,
    IdxVector&        recvTags,
    const Matrix&     sendVecs,
    const IdxVector&  sendTags )

{
  startSome ( EXCHANGE, sendTags );
  endSome   ( recvVecs, recvTags, sendVecs, sendTags );
}


//-----------------------------------------------------------------------
//   scatter
//-----------------------------------------------------------------------


inline void VectorExchanger::scatter ( const Vector& vec )
{
  startOne ( SCATTER );
  endOne   ( vec, vec );
}


inline void VectorExchanger::scatter

  ( const Vector&  recvVec,
    const Vector&  sendVec )

{
  startOne ( SCATTER );
  endOne   ( recvVec, sendVec );
}


inline void VectorExchanger::scatter

  ( Matrix&           recvVecs,
    IdxVector&        recvTags,
    const Matrix&     sendVecs,
    const IdxVector&  sendTags )

{
  startSome ( SCATTER,  sendTags );
  endSome   ( recvVecs, recvTags, sendVecs, sendTags );
}


//-----------------------------------------------------------------------
//   getMPContext
//-----------------------------------------------------------------------


inline Context* VectorExchanger::getMPContext () const
{
  return mpx_.get ();
}


//-----------------------------------------------------------------------
//   getDofSpace
//-----------------------------------------------------------------------


inline DofSpace* VectorExchanger::getDofSpace () const
{
  return dofs_.get ();
}


//-----------------------------------------------------------------------
//   getRecvBorders
//-----------------------------------------------------------------------


inline BorderSet VectorExchanger::getRecvBorders () const
{
  return recvBorders_;
}


//-----------------------------------------------------------------------
//   getSendBorders
//-----------------------------------------------------------------------


inline BorderSet VectorExchanger::getSendBorders () const
{
  return sendBorders_;
}


//-----------------------------------------------------------------------
//   hasOverlap
//-----------------------------------------------------------------------


inline bool VectorExchanger::hasOverlap () const
{
  return overlap_;
}


JIVE_END_PACKAGE( mp )

#endif

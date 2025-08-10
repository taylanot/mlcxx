/**
 * @file transform_impl.h
 * @author Ozgur Taylan Turan
 *
 * A simple transform wrapper, where you can apply mlpack transformations to the 
 * Dataset objects of our own. Here, you can also choose to apply those to 
 * inputs or the labels.
 */

#ifndef TRANSFORM_IMPL_H
#define TRANSFORM_IMPL_H

namespace data {
namespace regression {

//=============================================================================
// Transformer
//=============================================================================
template<class T, class D>
Transformer<T,D>::Transformer( const D& dataset )
{
  inp_.Fit(arma::conv_to<arma::mat>::from(dataset.inputs_));
  lab_.Fit(arma::conv_to<arma::rowvec>::from(dataset.labels_));
};

template<class T, class D>
D Transformer<T,D>::TransInp( const D& dataset )
{
  D tdataset = dataset;
  inp_.Transform( dataset.inputs_, tdataset.inputs_);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::TransLab ( const D& dataset )
{
  D tdataset = dataset;
  lab_.Transform( dataset.labels_, tdataset.labels_);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::Trans ( const D& dataset )
{
  D tdataset = TransInp(dataset);
  tdataset = TransLab(tdataset);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::InvTransInp( const D& dataset )
{
  D tdataset = dataset;
  inp_.InverseTransform( dataset.inputs_, tdataset.inputs_);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::InvTransLab ( const D& dataset )
{
  D tdataset = dataset;
  lab_.InverseTransform( dataset.labels_, tdataset.labels_);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::InvTrans ( const D& dataset )
{
  D tdataset = InvTransInp(dataset);
  return InvTransLab(tdataset);
};

} // regression namespace

namespace classification {

//=============================================================================
// Transformer
//=============================================================================
template<class T, class D>
Transformer<T,D>::Transformer( const D& dataset )
{
  trans_.Fit(dataset.inputs_);
};

template<class T, class D>
D Transformer<T,D>::Trans( const D& dataset )
{
  D tdataset = dataset;
  trans_.Transform(dataset.inputs_, tdataset.inputs_);
  return tdataset;
};

template<class T, class D>
D Transformer<T,D>::InvTrans( const D& dataset )
{
  D tdataset = dataset;
  trans_.InverseTransform( dataset.inputs_, tdataset.inputs_);
  return tdataset;
};

} // classification namespace

} // data namespace

#endif

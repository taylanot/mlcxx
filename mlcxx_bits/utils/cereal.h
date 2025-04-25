/**
 * @file cereal.h
 * @author Ozgur Taylan Turan
 *
 * Cereal is missing some of the objects so I will add them here.
 *
 */

#ifndef MY_CEREAL_H
#define MY_CEREAL_H


namespace cereal
{

	template <class Archive, class T>
	void save(Archive& ar, const std::optional<T>& opt)
	{
    bool hasVal = opt.has_value();
		ar(CEREAL_NVP(hasVal));
		if (hasVal)
		{
			const T& value = *opt;
			ar(cereal::make_nvp("value", value));
		}
	}

	template <class Archive, class T>
	void load(Archive& ar, std::optional<T>& opt)
	{
  	bool hasVal;
		ar(CEREAL_NVP(hasVal));
		if (hasVal)
		{
			T value;
			ar(CEREAL_NVP(value));
			opt = std::move(value);
		}
		else
			opt = std::nullopt;
		
	}
}
#endif

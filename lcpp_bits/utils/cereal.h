/**
 * @file cereal.h
 * @author Ozgur Taylan Turan
 *
 * Custom Cereal serialization helpers.
 * 
 * Cereal does not provide built-in support for some standard library types
 * such as std::optional and std::filesystem::path.
 * These save/load functions allow these types to be serialized and
 * deserialized in a portable way by converting them into simpler
 * serializable forms.
 *
 */

#ifndef MY_CEREAL_H
#define MY_CEREAL_H

namespace cereal {

//-----------------------------------------------------------------------------
// Save: std::optional<T> by first storing whether it has a value,
// then storing the value if present.
//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
// Load: std::optional<T> by reading presence flag, then value if present.
//-----------------------------------------------------------------------------
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
    {
        opt = std::nullopt;
    }
}

//-----------------------------------------------------------------------------
// Save: std::filesystem::path as a string.
//-----------------------------------------------------------------------------
template <class Archive>
void save(Archive& ar, const std::filesystem::path& path)
{
    ar(path.string());
}

//-----------------------------------------------------------------------------
// Load: std::filesystem::path from a stored string.
//-----------------------------------------------------------------------------
template <class Archive>
void load(Archive& ar, std::filesystem::path& path)
{
    std::string temp;
    ar(temp);
    path = std::filesystem::path(temp);
}

} //namespace cereal

#endif

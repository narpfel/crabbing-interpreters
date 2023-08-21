pub use variant_types_derive::derive_variant_types;

pub trait IntoVariant<Variant>
where
    Variant: IntoEnum<Enum = Self>,
{
    fn into_variant(self) -> Variant;
}

pub trait IntoEnum {
    type Enum;
    fn into_enum(self) -> Self::Enum;
}

impl<T> IntoEnum for &T
where
    T: IntoEnum + Copy,
{
    type Enum = <T as IntoEnum>::Enum;

    fn into_enum(self) -> Self::Enum {
        IntoEnum::into_enum(*self)
    }
}

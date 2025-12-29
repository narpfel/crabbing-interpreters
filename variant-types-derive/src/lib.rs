use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;
use syn::Data;
use syn::DeriveInput;
use syn::Error;
use syn::Fields;
use syn::FieldsNamed;
use syn::FieldsUnnamed;
use syn::Result;
use syn::Token;
use syn::TypePath;
use syn::Visibility;
use syn::fold::Fold;
use syn::parse_macro_input;
use syn::parse_quote;
use syn::spanned::Spanned;

struct ReferToTypesBySuper;
struct MakePublic;

impl Fold for ReferToTypesBySuper {
    fn fold_type_path(&mut self, type_path: TypePath) -> TypePath {
        let TypePath { qself, mut path } = type_path;
        let path = match path.leading_colon {
            Some(_) => path,
            None => {
                path.segments.insert(0, parse_quote!(super));
                path
            }
        };
        TypePath {
            qself: qself.map(|qself| self.fold_qself(qself)),
            path,
        }
    }
}

impl Fold for MakePublic {
    fn fold_visibility(&mut self, vis: Visibility) -> Visibility {
        Visibility::Public(Token![pub](vis.span()))
    }
}

#[proc_macro_attribute]
pub fn derive_variant_types(
    _attr: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive_variant_types_impl(input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn derive_variant_types_impl(input: DeriveInput) -> Result<TokenStream> {
    let vis = &input.vis;
    let enum_name = &input.ident;
    let enum_name = ReferToTypesBySuper.fold_type(parse_quote!(#enum_name));
    let name = format_ident!("{}Types", input.ident);
    let attrs = &input.attrs;

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let Data::Enum(type_) = &input.data
    else {
        return Err(Error::new_spanned(name, "must be an enum"));
    };

    let types = type_.variants.iter().map(|variant| {
        let name = &variant.ident;
        let fields = ReferToTypesBySuper.fold_fields(variant.fields.clone());
        let fields = MakePublic.fold_fields(fields);

        let structure = match &fields {
            Fields::Named(FieldsNamed { named, .. }) => {
                let fields = named.iter().map(|field| &field.ident);
                quote!(#name { #(#fields,)* })
            }
            Fields::Unnamed(FieldsUnnamed { unnamed, .. }) => {
                let fields = unnamed.iter().enumerate().map(|(i, field)| {
                    format_ident!(
                        "__{}_self_{}",
                        env!("CARGO_PKG_NAME").replace('-', "_"),
                        i,
                        span = field.span(),
                    )
                });
                quote!(#name(#(#fields,)*))
            }
            Fields::Unit => quote!(#name),
        };

        let semi = if let Fields::Named(_) = &variant.fields {
            None
        }
        else {
            Some(Token![;](variant.ident.span()))
        };
        quote::quote_spanned!(name.span() =>
            #(#attrs)*
            #vis struct #name #ty_generics #fields #semi

            impl #impl_generics ::variant_types::IntoEnum for #name #ty_generics #where_clause {
                type Enum = #enum_name #ty_generics;
                fn into_enum(self) -> Self::Enum {
                    let #structure = self;
                    Self::Enum::#structure
                }
            }

            impl #impl_generics ::variant_types::IntoVariant<#name #ty_generics>
            for #enum_name #ty_generics #where_clause {
                fn into_variant(self) -> #name #ty_generics {
                    let Self::#structure = self else { panic!() };
                    #structure
                }
            }

            impl #impl_generics #name #ty_generics #where_clause {
                pub fn loc(&self) -> super::Loc #ty_generics {
                    ::variant_types::IntoEnum::into_enum(self).loc()
                }

                pub fn slice(&self) -> &'a str {
                    ::variant_types::IntoEnum::into_enum(self).slice()
                }
            }
        )
    });

    Ok(quote::quote!(
        #input

        #[allow(non_snake_case)]
        #vis mod #name {
            #![allow(dead_code)]
            #(#types)*
        }
    ))
}

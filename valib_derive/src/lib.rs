use darling::{ast, FromDeriveInput, FromVariant};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};

#[proc_macro_derive(ParamName, attributes(name))]
pub fn derive_param_name(item: TokenStream) -> TokenStream {
    match DeriveParamName::from_derive_input(&syn::parse_macro_input!(item)) {
        Ok(d) => d.into_token_stream().into(),
        Err(error) => error.write_errors().into(),
    }
}

#[derive(Debug, FromVariant)]
#[darling(supports(unit), attributes(param_name))]
struct Variant {
    ident: syn::Ident,
    #[darling(rename = "display")]
    name: Option<String>,
}

impl Variant {
    fn impl_match_name(&self) -> proc_macro2::TokenStream {
        let Self { ident, name } = self;
        let name = name.clone().unwrap_or(ident.to_string());
        quote! {
            Self::#ident => std::borrow::Cow::Borrowed(#name)
        }
    }

    fn impl_from_id(&self, id: usize) -> proc_macro2::TokenStream {
        let Self { ident, .. } = self;
        let id = syn::Index::from(id);
        quote! {
            #id => Self::#ident
        }
    }

    fn impl_into_id(&self, id: usize) -> proc_macro2::TokenStream {
        let Self { ident, .. } = self;
        let id = syn::Index::from(id);
        quote! {
            Self::#ident => #id
        }
    }
}

#[derive(Debug, FromDeriveInput)]
#[darling(supports(enum_unit))]
struct DeriveParamName {
    ident: syn::Ident,
    data: ast::Data<Variant, ()>,
}

impl quote::ToTokens for DeriveParamName {
    fn to_tokens(&self, stream: &mut proc_macro2::TokenStream) {
        let Self { ident, data } = self;
        let ast::Data::Enum(fields) = data else {
            unreachable!();
        };
        let count = syn::Index::from(fields.len());
        let impl_name = fields.iter().map(|f| f.impl_match_name());
        let impl_intoid = fields.iter().enumerate().map(|(i, f)| f.impl_into_id(i));
        let impl_fromid = fields.iter().enumerate().map(|(i, f)| f.impl_from_id(i));
        let variants = fields
            .iter()
            .map(|Variant { ident, .. }| quote! { Self::#ident });
        stream.extend(quote! {
            impl ParamName for #ident {
                fn count() -> u64 {
                    #count
                }

                fn name(&self) -> std::borrow::Cow<'static, str> {
                    match self {
                        #(#impl_name),*
                    }
                }

                fn from_id(id: ParamId) -> Self {
                    match id {
                        #(#impl_fromid),*
                    }
                }

                fn into_id(self) -> Self {
                    match self {
                        #(#impl_intoid),*
                    }
                }

                fn iter() -> impl Iterator<Item=Self> {
                    [#(#variants),*].into_iter()
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_derive() {
        let input = syn::parse_str(
            /* rust */
            r#"enum DspParams {
                Cutoff,
                Resonance,
                Drive,
                #[param_name(display = "Input FM")]
                InputFM,
            }"#,
        )
        .expect("Parsing valid code");
        let from_derive_input =
            DeriveParamName::from_derive_input(&input).expect("Parsing valid code");
        eprintln!("{from_derive_input:#?}");
        let output = from_derive_input.into_token_stream().to_string();
        insta::assert_snapshot!(prettyplease::unparse(&syn::parse_file(&output).unwrap()));
    }
}

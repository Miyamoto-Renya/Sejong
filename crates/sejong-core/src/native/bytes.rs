use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ByteValue(pub Vec<u8>);
impl ByteValue {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}
impl From<String> for ByteValue {
    fn from(x: String) -> Self {
        Self(x.into_bytes())
    }
}
impl From<&str> for ByteValue {
    fn from(x: &str) -> Self {
        Self(x.as_bytes().to_vec())
    }
}
impl From<Vec<u8>> for ByteValue {
    fn from(x: Vec<u8>) -> Self {
        Self(x)
    }
}
impl AsRef<[u8]> for ByteValue {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl fmt::Display for ByteValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(text) = std::str::from_utf8(&self.0) {
            f.write_str(text)
        } else {
            f.write_str("0x")?;
            for b in &self.0 {
                write!(f, "{b:02x}")?;
            }
            Ok(())
        }
    }
}
impl Serialize for ByteValue {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        if let Ok(text) = std::str::from_utf8(&self.0) {
            s.serialize_str(text)
        } else {
            use serde::ser::SerializeMap;
            let mut map = s.serialize_map(Some(1))?;
            map.serialize_entry(
                "hex",
                &self
                    .0
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
            )?;
            map.end()
        }
    }
}
impl<'de> Deserialize<'de> for ByteValue {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Input {
            Text(String),
            Hex { hex: String },
        }
        match Input::deserialize(d)? {
            Input::Text(s) => Ok(s.into()),
            Input::Hex { hex } => {
                if !hex.len().is_multiple_of(2) || !hex.is_ascii() {
                    return Err(serde::de::Error::custom(
                        "even-length ASCII hexadecimal required",
                    ));
                }
                let bytes = hex
                    .as_bytes()
                    .chunks_exact(2)
                    .map(|pair| {
                        u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16)
                            .map_err(serde::de::Error::custom)
                    })
                    .collect::<Result<_, D::Error>>()?;
                Ok(Self(bytes))
            }
        }
    }
}

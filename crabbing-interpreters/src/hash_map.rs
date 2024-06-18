use std::hash::BuildHasher;
use std::hash::BuildHasherDefault;
use std::hash::Hash;

use rustc_hash::FxHasher;

enum IndexResult<'a, K, V> {
    Present(usize, &'a K, &'a V),
    NotPresent(usize),
}

#[derive(Debug, Clone)]
pub(crate) struct HashMap<K, V> {
    data: Vec<Option<(K, V)>>,
    length: usize,
    mask: usize,
    build_hasher: BuildHasherDefault<FxHasher>,
}

impl<K, V> HashMap<K, V>
where
    K: Hash + Eq,
{
    pub(crate) fn get(&self, key: &K) -> Option<&V> {
        match self.index(key) {
            IndexResult::Present(_index, _key, value) => Some(value),
            IndexResult::NotPresent(_) => None,
        }
    }

    fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        match self.index(key) {
            IndexResult::Present(index, _, _) => Some(&mut self.data[index].as_mut().unwrap().1),
            IndexResult::NotPresent(_) => None,
        }
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        if let Some(v) = self.get_mut(&key) {
            *v = value;
        }
        else {
            if self.load_factor() >= 0.8 {
                self.mask = ((self.mask * 2) | 1).max(0b111);
                let data = std::mem::take(&mut self.data);
                self.data.resize_with(self.capacity(), || None);
                for (k, v) in data.into_iter().flatten() {
                    match self.index(&k) {
                        IndexResult::Present(_, _, _) => unreachable!(),
                        IndexResult::NotPresent(index) => self.data[index] = Some((k, v)),
                    }
                }
            }
            match self.index(&key) {
                IndexResult::Present(index, _, _) | IndexResult::NotPresent(index) =>
                    self.data[index] = Some((key, value)),
            }
            self.length += 1;
        }
    }

    fn index(&self, key: &K) -> IndexResult<K, V> {
        #[expect(clippy::as_conversions)]
        let hash = self.build_hasher.hash_one(key) as usize;
        let mut index = hash & self.mask;
        while let Some(Some((k, v))) = self.data.get(index) {
            if k == key {
                return IndexResult::Present(index, k, v);
            }
            index += 1;
            index &= self.mask;
        }
        IndexResult::NotPresent(index)
    }
}

impl<K, V> HashMap<K, V> {
    fn capacity(&self) -> usize {
        self.mask + 1
    }

    fn load_factor(&self) -> f64 {
        #[expect(clippy::as_conversions)]
        ((self.length as f64 + 1.0) / (self.capacity() as f64))
    }

    pub(crate) fn values(&self) -> impl Iterator<Item = &V> {
        self.data
            .iter()
            .flat_map(|slot| slot.as_ref().map(|(_k, v)| v))
    }
}

impl<K, V> Default for HashMap<K, V> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            length: 0,
            mask: 0,
            build_hasher: BuildHasherDefault::default(),
        }
    }
}

impl<K, V> FromIterator<(K, V)> for HashMap<K, V>
where
    K: Hash + Eq,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = Self::default();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

use std::rc::Rc;

use anyhow::Result;
use ash::Entry;

pub struct EntryObject {
    entry: Rc<Entry>,
}
impl EntryObject {
    /// EntryObjectを作成する。
    pub fn new() -> Result<Self> {
        let entry = Entry::new()?;
        Ok(Self {
            entry: Rc::new(entry),
        })
    }

    /// Entryを取得する。
    pub fn entry(&self) -> Rc<Entry> {
        Rc::clone(&self.entry)
    }

    /// Entryの参照を取得する。
    pub fn entry_as_ref(&self) -> &Entry {
        self.entry.as_ref()
    }
}

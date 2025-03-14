use std::{
    iter::FusedIterator,
    ops::{Deref, DerefMut},
};

use crate::{
    piece::{Piece, PieceType},
    square::Square,
};

pub const SQUARE_LIST_SIZE: usize = 10;

/// Invariant: All elements up to `size` must be initialized
#[derive(Debug, Clone, Copy)]
pub struct SquareList {
    list: [Square; SQUARE_LIST_SIZE],
    size: usize,
}

impl SquareList {
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub const fn new() -> Self {
        Self {
            list: [Square::from_square(0); SQUARE_LIST_SIZE],
            size: 0,
        }
    }

    /// Adds a square to the list
    ///
    /// # Panics
    ///
    /// Panics if the list is already full
    pub fn add(&mut self, sq: Square) {
        if self.contains(sq) {
            return;
        }
        assert!(
            self.size != SQUARE_LIST_SIZE,
            "Tried to add a square to a SquareList over the allowed defined maximum SQUARE_LIST_SIZE ({SQUARE_LIST_SIZE})"
        );
        self.size += 1;
        self.list[self.size - 1] = sq;
    }

    /// Removes a square from the list
    /// Returns a bool wether it was successful or not
    pub fn remove(&mut self, square_to_be_removed: Square) -> bool {
        if let Some(position_to_be_removed) = self.iter().position(|sq| sq == &square_to_be_removed)
        {
            self.list.swap(position_to_be_removed, self.size - 1);
            self.size -= 1;
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn contains(&self, sq: Square) -> bool {
        self.iter().any(|x| x == &sq)
    }

    #[must_use]
    pub const fn into_iter_as_piece(
        self,
        piece_type: PieceType,
        is_white: bool,
    ) -> PieceListIterator {
        PieceListIterator {
            l_index: 0,
            r_index: self.size,
            piece_type,
            is_white,
            list: self.list,
        }
    }
}

impl Default for SquareList {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for SquareList {
    type Target = [Square];

    fn deref(&self) -> &Self::Target {
        &self.list[..self.size]
    }
}

impl DerefMut for SquareList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.list[..self.size]
    }
}

impl Eq for SquareList {}

impl PartialEq for SquareList {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self[..] == other[..]
    }
}

pub struct PieceListIterator {
    l_index: usize,
    r_index: usize,
    piece_type: PieceType,
    is_white: bool,
    list: [Square; SQUARE_LIST_SIZE],
}

impl ExactSizeIterator for PieceListIterator {}
impl FusedIterator for PieceListIterator {}

impl Iterator for PieceListIterator {
    type Item = Piece;

    fn next(&mut self) -> Option<Self::Item> {
        if self.r_index <= self.l_index {
            return None;
        }
        let val = self.list[self.l_index];
        self.l_index += 1;
        Some(Piece::new(val, self.piece_type, self.is_white))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.r_index - self.l_index;
        (size, Some(size))
    }
}

impl DoubleEndedIterator for PieceListIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.r_index <= self.l_index {
            return None;
        }
        self.r_index -= 1;
        let val = self.list[self.r_index];
        Some(Piece::new(val, self.piece_type, self.is_white))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(
        expected = "Tried to add a square to a SquareList over the allowed defined maximum SQUARE_LIST_SIZE"
    )]
    fn squarelist_panic_on_too_many_elements() {
        let mut squarelist = SquareList::new();
        for i in
            0..=u8::try_from(SQUARE_LIST_SIZE).expect("SQUARE_LIST_SIZE should not exceed u8::MAX")
        {
            squarelist.add(Square::from_square(i));
        }
    }

    #[test]
    fn squarelist_add_3_remove_1() {
        let mut squarelist = SquareList::new();
        squarelist.add(Square::from_lateral(2, 3)); // c4
        squarelist.add(Square::from_lateral(1, 3)); // b4
        squarelist.add(Square::from_lateral(3, 3)); // d4

        assert_eq!(squarelist[0], Square::from_lateral(2, 3));
        assert_eq!(squarelist[1], Square::from_lateral(1, 3));
        assert_eq!(squarelist[2], Square::from_lateral(3, 3));

        squarelist.remove(Square::from_lateral(1, 3));
        assert_eq!(squarelist[0], Square::from_lateral(2, 3));
        assert_eq!(squarelist[1], Square::from_lateral(3, 3));
    }

    #[test]
    fn squarelist_exactsizeiterator() {
        let squarelist = SquareList {
            list: [Square::from_square(0); SQUARE_LIST_SIZE],
            size: 3,
        };

        let mut iter = squarelist.iter();
        assert_eq!(iter.len(), iter.size_hint().0);
        assert_eq!(Some(iter.len()), iter.size_hint().1);
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next_back();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn squarelist_doubleendediterator() {
        let mut squarelist = SquareList::new();
        for i in 0..=3u8 {
            squarelist.add(Square::from_square(i));
        }

        let mut iter = squarelist.iter();
        assert_eq!(iter.next(), Some(&Square::from_square(0)));
        assert_eq!(iter.next_back(), Some(&Square::from_square(3)));
        assert_eq!(iter.next(), Some(&Square::from_square(1)));
        assert_eq!(iter.next_back(), Some(&Square::from_square(2)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn squarelist_contains() {
        let mut squarelist = SquareList::new();
        for i in 0..=3u8 {
            squarelist.add(Square::from_square(i));
        }

        assert!(squarelist.contains(Square::from_square(2)));
        assert!(!squarelist.contains(Square::from_square(9)));
    }
}

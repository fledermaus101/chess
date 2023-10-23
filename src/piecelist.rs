use std::{
    iter::FusedIterator,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
};

use crate::{Piece, PieceType, Square};

pub(crate) const PIECE_LIST_SIZE: usize = 10;

/// Invariant: All elements up to `size` must be initialized
#[derive(Debug, Clone, Copy)]
pub(crate) struct InternalPieceList {
    list: [MaybeUninit<Square>; PIECE_LIST_SIZE],
    size: usize,
}

impl InternalPieceList {
    pub(crate) const fn new() -> InternalPieceList {
        InternalPieceList {
            list: MaybeUninit::uninit_array(), // will be overwritten anyways
            size: 0,
        }
    }

    pub(crate) const fn as_piece_list(self, piece_type: PieceType, is_white: bool) -> PieceList {
        PieceList {
            internal_list: self,
            piece_type,
            is_white,
        }
    }

    pub(crate) fn add(&mut self, sq: Square) {
        if self[..].iter().any(|x| x == &sq) {
            return;
        }
        assert!(
            self.size != PIECE_LIST_SIZE,
            "Tried to add a square to a InternalPieceList over the allowed defined maximum PIECE_LIST_SIZE ({PIECE_LIST_SIZE})"
        );
        self.size += 1;
        self.list[self.size - 1].write(sq);
    }

    /// Removes a square from the list
    /// Returns a bool wether it was successful or not
    pub(crate) fn remove(&mut self, square_to_be_removed: Square) -> bool {
        if let Some(position_to_be_removed) =
            self[..].iter().position(|sq| sq == &square_to_be_removed)
        {
            if self.size != 1 {
                self.list.swap(position_to_be_removed, self.size - 1);
            }
            self.size -= 1;
            true
        } else {
            false
        }
    }

    pub(crate) fn contains(&self, sq: Square) -> bool {
        self[..].iter().any(|x| x == &sq)
    }
}

impl Deref for InternalPieceList {
    type Target = [Square];

    fn deref(&self) -> &Self::Target {
        // SAFETY: All elements of the list up to self.size must always be initialized
        unsafe { MaybeUninit::slice_assume_init_ref(&self.list[..self.size]) }
    }
}

impl DerefMut for InternalPieceList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: see impl Deref
        unsafe { MaybeUninit::slice_assume_init_mut(&mut self.list[..self.size]) }
    }
}

impl Eq for InternalPieceList {}

impl PartialEq for InternalPieceList {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size && self[..] == other[..]
    }
}

impl IntoIterator for InternalPieceList {
    type Item = Square;

    type IntoIter = InternalPieceListIterator;

    fn into_iter(self) -> Self::IntoIter {
        InternalPieceListIterator {
            l_index: 0,
            r_index: self.size,
            list: self.list,
        }
    }
}

pub(crate) struct InternalPieceListIterator {
    l_index: usize,
    r_index: usize,
    list: [MaybeUninit<Square>; PIECE_LIST_SIZE],
}

impl ExactSizeIterator for InternalPieceListIterator {}
impl FusedIterator for InternalPieceListIterator {}

impl Iterator for InternalPieceListIterator {
    type Item = Square;

    fn next(&mut self) -> Option<Self::Item> {
        if self.r_index <= self.l_index {
            return None;
        }
        let val = self.list[self.l_index];
        self.l_index += 1;
        Some(unsafe { val.assume_init() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.r_index - self.l_index;
        (size, Some(size))
    }
}

impl DoubleEndedIterator for InternalPieceListIterator {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.r_index <= self.l_index {
            return None;
        }
        self.r_index -= 1;
        let val = self.list[self.r_index];
        Some(unsafe { val.assume_init() })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PieceList {
    internal_list: InternalPieceList,
    piece_type: PieceType,
    is_white: bool,
}

impl PieceList {
    pub fn len(&self) -> usize {
        self.internal_list.size
    }

    /// Checks if this piecelist contains a piece with this specific square
    pub fn contains_square(&self, sq: Square) -> bool {
        self.internal_list.contains(sq)
    }

    /// Checks if this piecelist contains a piece
    pub fn contains(&self, other_piece: Piece) -> bool {
        self.piece_type() == other_piece.piece_type()
            && self.is_white() == other_piece.is_white()
            && self.contains_square(other_piece.square())
    }

    pub fn piece_type(&self) -> PieceType {
        self.piece_type
    }

    pub fn is_white(&self) -> bool {
        self.is_white
    }
}

impl IntoIterator for PieceList {
    type Item = crate::Piece;

    type IntoIter = PieceListIterator;

    fn into_iter(self) -> Self::IntoIter {
        PieceListIterator {
            piece_type: self.piece_type,
            is_white: self.is_white,
            internal_list_iterator: self.internal_list.into_iter(),
        }
    }
}

pub struct PieceListIterator {
    piece_type: PieceType,
    is_white: bool,
    internal_list_iterator: InternalPieceListIterator,
}

impl ExactSizeIterator for PieceListIterator {}
impl FusedIterator for PieceListIterator {}

impl Iterator for PieceListIterator {
    type Item = Piece;

    fn next(&mut self) -> Option<Self::Item> {
        self.internal_list_iterator.next().map(|sq| Piece {
            piece_type: self.piece_type,
            is_white: self.is_white,
            square: sq,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.internal_list_iterator.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(
        expected = "Tried to add a square to a InternalPieceList over the allowed defined maximum PIECE_LIST_SIZE"
    )]
    fn piecelist_panic_on_too_many_elements() {
        let mut piecelist = InternalPieceList::new();
        for i in 0..(PIECE_LIST_SIZE + 1) {
            piecelist.add(Square::from_square(i as u8));
        }
    }

    #[test]
    fn piecelist_add_3_remove_1() {
        let mut piecelist = InternalPieceList::new();
        piecelist.add(Square::from_lateral(2, 3)); // c4
        piecelist.add(Square::from_lateral(1, 3)); // b4
        piecelist.add(Square::from_lateral(3, 3)); // d4

        assert_eq!(piecelist[0], Square::from_lateral(2, 3));
        assert_eq!(piecelist[1], Square::from_lateral(1, 3));
        assert_eq!(piecelist[2], Square::from_lateral(3, 3));

        piecelist.remove(Square::from_lateral(1, 3));
        assert_eq!(piecelist[0], Square::from_lateral(2, 3));
        assert_eq!(piecelist[1], Square::from_lateral(3, 3));
    }

    #[test]
    fn piecelist_exactsizeiterator() {
        let piecelist = InternalPieceList {
            // Can't use MaybeUninit::zeroed() because Square isn't repr(transparent)
            list: [Square::from_square(0); PIECE_LIST_SIZE].map(MaybeUninit::new),
            size: 3,
        };
        let mut iter = piecelist.into_iter();
        assert_eq!(iter.len(), iter.size_hint().0);
        assert_eq!(iter.len(), iter.size_hint().1.unwrap());
        assert_eq!(iter.len(), 3);
        iter.next();
        assert_eq!(iter.len(), 2);
        iter.next_back();
        assert_eq!(iter.len(), 1);
        iter.next();
        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn piecelist_doubleendediterator() {
        let mut piecelist = InternalPieceList::new();
        for i in 0..=3 {
            piecelist.add(Square::from_square(i as u8));
        }
        let mut iter = piecelist.into_iter();
        assert_eq!(iter.next(), Some(Square::from_square(0)));
        assert_eq!(iter.next_back(), Some(Square::from_square(3)));
        assert_eq!(iter.next(), Some(Square::from_square(1)));
        assert_eq!(iter.next_back(), Some(Square::from_square(2)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    #[test]
    fn piecelist_contains() {
        let mut piecelist_internal = InternalPieceList::new();
        for i in 0..=3 {
            piecelist_internal.add(Square::from_square(i as u8));
        }

        assert!(piecelist_internal.contains(Square::from_square(2)));
        assert!(!piecelist_internal.contains(Square::from_square(9)));

        let piecelist = piecelist_internal.as_piece_list(PieceType::Pawn, false);
        assert!(piecelist.contains(Piece::new(Square::from_square(1), PieceType::Pawn, false)));
        assert!(!piecelist.contains(Piece::new(Square::from_square(1), PieceType::Pawn, true)));
        assert!(!piecelist.contains(Piece::new(Square::from_square(1), PieceType::Queen, false)));
        assert!(!piecelist.contains(Piece::new(Square::from_square(8), PieceType::Pawn, false)));

        assert!(piecelist.contains_square(Square::from_square(2)));
        assert!(!piecelist.contains_square(Square::from_square(9)));
    }
}

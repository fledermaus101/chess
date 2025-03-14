use std::{
    fmt::Display,
    ops::{Add, Sub},
};

use thiserror::Error;

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct Square(u8);

impl Add for Square {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 + rhs.0).expect(
            "(Square + Square) was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Add<(i8, i8)> for Square {
    type Output = Self;

    fn add(self, rhs: (i8, i8)) -> Self::Output {
        self.try_add_tuple(rhs).expect(
            "(Square + tuple) was expected to be inside the chess board, but was over the maximum",
        )
    }
}

impl Sub for Square {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::try_from_square(self.0 - rhs.0).expect(
            "(Square - Square) was expected to be inside the chess board, but was under the minimum",
        )
    }
}

impl Sub<(i8, i8)> for Square {
    type Output = Self;

    fn sub(self, rhs: (i8, i8)) -> Self::Output {
        let (file, rank) = rhs;
        self.try_add_tuple((-file, -rank))
            .expect("(Square - tuple) was expected to be inside the chess board, but wasn't")
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Error)]
pub enum AlgebraicSqaureConversionError {
    #[error("String length was expected to be 2. Got {0}")]
    WrongLength(usize),
    #[error("Expected file char to be one of a, b, c, d, e, f, g, h. Got '{0}'")]
    InvalidFileChar(char),
    #[error("Expected rank char to be a digit. Got '{0}'")]
    RankCharIsNotADigit(char),
    #[error("Expected rank char to be a digit in the inclusive range from 1 to 8. Got {0}")]
    RankDigitIsOutsideOfValidRange(u32),
}

#[derive(Clone, Eq, PartialEq, Debug, Error)]
pub enum LateralPositionToSquareConversionError {
    #[error("file (0 indexed) should not be larger than 7. Actual {0}")]
    FileTooLarge(u8),
    #[error("rank (0 indexed) should not be larger than 7. Actual {0}")]
    RankTooLarge(u8),
    #[error("file should not be negative. Actual {0}")]
    FileNegative(i8),
    #[error("rank should not be negative. Actual {0}")]
    RankNegative(i8),
}

impl Display for Square {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", (self.file() + b'a') as char, self.rank() + 1)
    }
}

impl Square {
    /// Returns a square given a flattened coordinate
    ///
    /// # Panics
    ///
    /// Panics if `square_index > 63`
    #[must_use]
    pub const fn from_square(square_index: u8) -> Self {
        Self::try_from_square(square_index)
            .expect("Square was expected to be inside the chess board, but was over the maximum")
    }

    /// Returns a square given a flattened coordinate if `square_index <= 63`
    #[must_use]
    pub const fn try_from_square(square_index: u8) -> Option<Self> {
        if square_index <= 63 {
            Some(Self(square_index))
        } else {
            None
        }
    }

    /// Returns a square given lateral coordinates
    ///
    /// # Errors
    ///
    /// This function will return an error if `file > 7 || rank > 7`
    pub const fn try_from_lateral(
        file: u8,
        rank: u8,
    ) -> Result<Self, LateralPositionToSquareConversionError> {
        if file > 7 {
            return Err(LateralPositionToSquareConversionError::FileTooLarge(file));
        }
        if rank > 7 {
            return Err(LateralPositionToSquareConversionError::RankTooLarge(rank));
        }
        Ok(Self(rank * 8 + file))
    }

    /// Returns a square given lateral coordinates
    ///
    /// # Panics
    ///
    /// Panics if `file > 7 || rank > 7`
    #[must_use]
    pub const fn from_lateral(file: u8, rank: u8) -> Self {
        assert!(file <= 7, "file cannot be larger than 7 (0 indexed)");
        assert!(rank <= 7, "rank cannot be larger than 7 (0 indexed)");
        Self(rank * 8 + file)
    }

    #[must_use]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    #[must_use]
    pub const fn square(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn to_tuple(self) -> (u8, u8) {
        (self.file(), self.rank())
    }

    /// Returns a new square with the added file offset
    ///
    /// # Panics
    ///
    /// Panics if [`file`](Self) + offset is not inside `0..=7`
    #[must_use]
    pub fn add_file(self, offset: i8) -> Self {
        self.try_add_tuple((offset, 0))
            .expect("Cannot add file offset")
    }

    /// Returns a new square with the added rank offset
    ///
    /// # Panics
    ///
    /// Panics if `[rank](Self) + offset` is not inside `0..=7`
    #[must_use]
    pub fn add_rank(self, offset: i8) -> Self {
        self.try_add_tuple((0, offset))
            .expect("Cannot add rank offset")
    }

    #[must_use]
    pub const fn try_add(self, rhs: Self) -> Option<Self> {
        Self::try_from_square(self.0 + rhs.0)
    }

    /// Adds the square and tuple together
    ///
    /// # Errors
    ///
    /// This function will return an error if the resulting square is out of bounds, meaning one of
    /// these conditions is true
    /// - `file` is not inside `0..=7`
    /// - `rank` is not inside `0..=7`
    pub const fn try_add_tuple(
        self,
        rhs: (i8, i8),
    ) -> Result<Self, LateralPositionToSquareConversionError> {
        let (lhs_file, lhs_rank) = self.to_tuple();
        let (rhs_file, rhs_rank) = rhs;
        let file = match lhs_file.checked_add_signed(rhs_file) {
            Some(f) => Ok(f),
            None => Err(LateralPositionToSquareConversionError::FileNegative(
                rhs_file.saturating_add_unsigned(lhs_file),
            )),
        };
        let rank = match lhs_rank.checked_add_signed(rhs_rank) {
            Some(r) => Ok(r),
            None => Err(LateralPositionToSquareConversionError::RankNegative(
                rhs_rank.saturating_add_unsigned(lhs_rank),
            )),
        };
        match (file, rank) {
            (Ok(file), Ok(rank)) => Self::try_from_lateral(file, rank),
            (Ok(_), Err(err)) | (Err(err), _) => Err(err),
        }
    }

    /// Returns a square given a square name
    ///
    /// # Panics
    ///
    /// See [`try_from_algebraic`](Self)
    #[must_use]
    pub fn from_algebraic(square: &str) -> Self {
        Self::try_from_algebraic(square).expect("Square was not able to be parsed.")
    }

    /// Returns a square given a square name
    ///
    /// # Errors
    ///
    /// This function will return an error if
    /// - `square[0]` is not one of `'a'..='h'`
    /// - `square[1]` is not one of `'1'..='8'`
    #[allow(clippy::missing_panics_doc)]
    pub fn try_from_algebraic(square: &str) -> Result<Self, AlgebraicSqaureConversionError> {
        if square.len() != 2 {
            return Err(AlgebraicSqaureConversionError::WrongLength(square.len()));
        }

        let mut chars = square.chars();
        let file = match chars.next().expect(
            "String square is empty. This should be impossible because of the length check.",
        ) {
            ch @ 'a'..='h' => ch as u8 - b'a',
            ch => return Err(AlgebraicSqaureConversionError::InvalidFileChar(ch)),
        };

        let rank = chars.next().expect("String is only one char long. This should be impossible because of the length check").to_digit(10).expect("Expected rank to be a digit");
        if !(1..=8).contains(&rank) {
            return Err(AlgebraicSqaureConversionError::RankDigitIsOutsideOfValidRange(rank));
        }
        Ok(Self::from_lateral(
            file,
            u8::try_from(rank - 1).expect("Above check should make this impossible"),
        ))
    }

    #[must_use]
    pub const fn rank_flip(self) -> Self {
        Self::from_lateral(self.file(), 7 - self.rank())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_add_tuple() {
        assert_eq!(
            Square::from_lateral(2, 3) + (5, 4),
            Square::from_lateral(7, 7)
        );
    }

    #[test]
    fn square_add_tuple_negative() {
        assert_eq!(
            Square::from_lateral(2, 2).try_add_tuple((-3, -1)),
            Err(LateralPositionToSquareConversionError::FileNegative(-1))
        );
        assert_eq!(
            Square::from_lateral(2, 2).try_add_tuple((-1, -3)),
            Err(LateralPositionToSquareConversionError::RankNegative(-1))
        );
        assert_eq!(
            Square::from_lateral(6, 3) + (-6, -2),
            Square::from_lateral(0, 1)
        );
    }

    #[test]
    fn square_add_square() {
        assert_eq!(
            Square::from_lateral(3, 6) + Square::from_lateral(4, 1),
            Square::from_lateral(7, 7)
        );
    }

    #[test]
    fn square_add_square_over_max() {
        assert_eq!(
            Square::from_lateral(4, 6).try_add(Square::from_lateral(4, 1)),
            None
        );
    }
}

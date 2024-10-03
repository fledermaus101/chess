use chess::squarelist::SquareList;
use chess::Square;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add 10 elems", |b| {
        b.iter(|| {
            let mut piecelist = SquareList::new();
            for i in 0..10 {
                piecelist.add(black_box(Square::from_square(i)));
            }
        });
    });
    c.bench_function("remove all elems", |b| {
        b.iter(|| {
            let mut piecelist = SquareList::new();
            for i in 0..10 {
                piecelist.add(black_box(Square::from_square(i)));
            }
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

use chess::piecelist::SquareList;
use chess::Square;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("add 10 elems", |b| {
        b.iter(|| {
            let mut piecelist = SquareList::new();
            for i in 0..10 {
                piecelist.add(black_box(Square::from_square(i)));
            }
        })
    });
    c.bench_function("remove all elems", |b| {
        b.iter(|| {
            let mut piecelist = SquareList::new();
            for i in 0..10 {
                piecelist.add(black_box(Square::from_square(i)));
            }
        })
    });

    let mut group = c.benchmark_group("Complements");
    for i in [0, 3 * 8 + 1, 4 * 8 + 6, 7 * 8 + 7] {
        let square = Square::from_square(i);
        group.bench_with_input(
            BenchmarkId::new("Convert to lateral and back", i),
            &square,
            |b, sq| b.iter(|| Square::from_lateral(7 - sq.rank(), sq.file())),
        );
        group.bench_with_input(
            BenchmarkId::new("Alternate version with only one modulo", i),
            &square,
            |b, sq| b.iter(|| sq.complement_rank()),
        );
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

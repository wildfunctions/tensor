pub mod tensor;
pub use tensor::{Rank1Tensor, Rank2Tensor, Tensor};

fn main() {
    //let x1 = vec![1,0];
    //let x2 = vec![0,1];

    //let v1 = Rank1Tensor::build(2, x1);
    //let v2 = Rank1Tensor::build(2, x2); 

    //let t1 = v1 * v2;

    //let a1 = vec![1,0];
    //let a2 = vec![0,1];
    
    //3-d Rank2 Tensor
    let t2 = Tensor::build(3, 2, vec![
        1, 2, 3, 
        4, 5, 6, 
        7, 8, 9
    ]);
    
    //4-d Rank1 Tensor
    let t1 = Tensor::build(4, 1, vec![
        2, 3, 5, 5
    ]); 

    //3d Rank3 Tensor
    let t3 = Tensor::build(3, 3, vec![
        1, 2, 3,
        4, 5, 6,
        7, 8, 9, 
       
        10, 11, 12,
        13, 14, 15,
        16, 17, 18,

        19, 20, 21,
        22, 23, 24,
        25, 26, 27 
    ]);
       

    //2-d Rank2 Tensor
    let t4 = Tensor::build(2, 2, vec![
        1, 2, 
        3, 4
    ]);

    //2-d Rank3 Tensor
    let t5 = Tensor::build(2, 3, vec![
        1, 2,
        3, 4,
 
        5, 6,
        7, 8
    ]); 

    println!( "Printing t1" );
    t1.print(); 

    println!( "Printing t2" );
    t2.print();

    println!( "Printing t3" );
    t3.print();

    println!( "Printing t4" );
    t4.print();

    println!( "Printing t5" );
    t5.print();
}

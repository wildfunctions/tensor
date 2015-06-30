use traits;
use tensor::Tensor;

//tmp solution for n^p 
pub fn scalar_power(n: i32, p: i32) -> i32 {
    let mut a: i32 = 1;
    for _ in 0..p { a = a*n; } 
    a
}

//entry point for inner_product_loop_many
pub fn inner_product_loop<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&mut Tensor<T>, &Tensor<T>, &Tensor<T>, &[i32]),
t1: &Tensor<T>, t2: &Tensor<T>, t3: &mut Tensor<T>) {
    inner_product_loop_many(max_indices.clone(), f, t1, t2, t3, Vec::new(), 0);
}

//variable depth inner product loop
pub fn inner_product_loop_many<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&mut Tensor<T>, &Tensor<T>, &Tensor<T>, &[i32]), 
t1: &Tensor<T>, t2: &Tensor<T>, t3: &mut Tensor<T>, pargs: Vec<i32>, index: i32) {
    if max_indices.len() == 0 {
        f(t3, t1, t2, &pargs); 
    } else {
        let mut args = pargs.clone();
        let rest: Vec<i32> = max_indices[1..].to_vec();
        for _ in 0..max_indices[0] {
            if args.len() == index as usize { args.push(0); }
            if args[index as usize] < max_indices[0] {
                inner_product_loop_many(rest.clone(), f, t1, t2, t3, args.clone(), index + 1);
                args[index as usize] = args[index as usize] + 1;
            }
        }
    }
}

//entry point for print_loop_many
pub fn print_loop<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&Tensor<T>, Vec<i32>), t: &Tensor<T>) {
    print_loop_many(max_indices.clone(), f, t, Vec::new(), 0);
}

//variable depth print loop
pub fn print_loop_many<T: traits::TensorTrait<T>>(max_indices: Vec<i32>, f: fn(&Tensor<T>, Vec<i32>), 
t: &Tensor<T>, pargs: Vec<i32>, index: i32) {
    if max_indices.len() == 0 {
        f(t, pargs); 
    } else {
        let mut args = pargs.clone();
        let rest: Vec<i32> = max_indices[1..].to_vec();
        for _ in 0..max_indices[0] {
            if args.len() == index as usize { args.push(0); }
            if args[index as usize] < max_indices[0] {
                print_loop_many(rest.clone(), f, t, args.clone(), index + 1);
                args[index as usize] = args[index as usize] + 1;
            }
        }
    }
}

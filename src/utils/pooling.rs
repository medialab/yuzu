use ndarray::{Array2, ArrayViewD, Ix2, s};

fn print_shape3(last_hidden_state: &ArrayViewD<f32>) {
    let (batch_size, seq_len, hidden_dim) = {
        let shape = last_hidden_state.shape();
        (shape[0], shape[1], shape[2])
    };

    println!(
        "batch size: {}, seq_len: {}, hidden_dim: {}",
        batch_size, seq_len, hidden_dim
    );
}

pub fn mean_pooling(
    // Inspired from https://docs.rs/fastembed/latest/src/fastembed/pooling.rs.html#34
    last_hidden_state: &ArrayViewD<f32>,
    attention_mask_array: &ArrayViewD<i64>,
) -> Array2<f32> {
    print_shape3(last_hidden_state);

    let token_embeddings = last_hidden_state.slice(s![.., .., ..]);
    let attention_mask = attention_mask_array
        .clone()
        .insert_axis(ndarray::Axis(2))
        .broadcast(token_embeddings.dim())
        .unwrap()
        .mapv(|x| x as f32);

    let masked_tensor = &attention_mask * &token_embeddings;
    let sum = masked_tensor.sum_axis(ndarray::Axis(1));
    let mask_sum = attention_mask.sum_axis(ndarray::Axis(1));
    let mask_sum = mask_sum.mapv(|x| if x == 0f32 { 1.0 } else { x });
    sum / mask_sum
}

pub fn last_token(last_hidden_state: &ArrayViewD<f32>) -> Array2<f32> {
    print_shape3(last_hidden_state);

    let token_embeddings = last_hidden_state.slice(s![.., .., ..]);
    let sliced = token_embeddings.slice(s![.., -1, ..]);
    let last_token = sliced.to_owned().into_dimensionality::<Ix2>().unwrap();

    last_token
}
